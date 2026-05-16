"""On-demand config change detection via SHA-256 hashing.

No background threads, no fs-notify.  Callers poll
:meth:`OrchidConfigWatcherBase.has_changes` (or the convenience
:meth:`reload_if_changed`) when they are ready to pay the stat cost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .frontmatter import compute_sha256, load_markdown_file
from .loader import load_config as _load_yaml_config
from .md_loader import _merge_agent_md, load_md_config
from .schema import OrchidAgentsConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Snapshot
# ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrchidConfigSnapshot:
    """Immutable snapshot of the current config plus file hashes.

    Attributes
    ----------
    config : OrchidAgentsConfig
        Fully-validated configuration.
    file_hashes : dict[str, str]
        Absolute file-path string → hex SHA-256 for every file that was
        read to produce *config*.
    root_path : Path
        The root config file (``orchid.md`` or ``orchid.yml``).
    """

    config: OrchidAgentsConfig
    file_hashes: dict[str, str]
    root_path: Path


# ──────────────────────────────────────────────────────────────────
# Base Watcher (shared change-detection logic)
# ──────────────────────────────────────────────────────────────────


class OrchidConfigWatcherBase(ABC):
    """Abstract base for on-demand config change detection.

    Subclasses implement :meth:`_reload` (full reload) and
    :meth:`_reload_single_agent` (per-agent reload) — the hashing,
    comparison, and snapshot management are shared here.
    """

    def __init__(
        self,
        *,
        initial_config: OrchidAgentsConfig,
        initial_hashes: dict[str, str],
        root_path: Path,
    ) -> None:
        self._snapshot = OrchidConfigSnapshot(
            config=initial_config,
            file_hashes=dict(initial_hashes),
            root_path=root_path,
        )

    # ── Read-only ──────────────────────────────────────────

    @property
    def snapshot(self) -> OrchidConfigSnapshot:
        return self._snapshot

    # ── Subclass hooks ─────────────────────────────────────

    @abstractmethod
    def _current_hashes(self) -> dict[str, str]:
        """Recompute SHA-256 for every tracked file plus any new entries."""

    @abstractmethod
    def _reload(self) -> OrchidConfigSnapshot:
        """Force a full reload from disk and return the new snapshot."""

    @abstractmethod
    def _reload_single_agent(self, name: str) -> OrchidConfigSnapshot | None:
        """Reload a single agent and merge it into the current config."""

    # ── Change detection (shared) ──────────────────────────

    def has_changes(self) -> bool:
        """Return ``True`` when any tracked file has been added,
        deleted, or modified since the last snapshot."""
        current = self._current_hashes()
        if not current:
            return True
        if set(current.keys()) != set(self._snapshot.file_hashes.keys()):
            return True
        return any(current[k] != self._snapshot.file_hashes[k] for k in current)

    def changed_files(self) -> list[str]:
        """Return the paths of files that differ from the snapshot.

        Includes newly-created files that were not in the previous
        snapshot.  When the root file is deleted, ``has_changes()``
        returns ``True`` but this method returns ``[]`` — the caller
        already knows something changed.
        """
        current = self._current_hashes()
        changed: list[str] = []
        if not current:
            return changed

        for path_str, new_hash in current.items():
            old_hash = self._snapshot.file_hashes.get(path_str)
            if old_hash is None or old_hash != new_hash:
                changed.append(path_str)
        return changed

    # ── Reload convenience (shared) ────────────────────────

    def reload_if_changed(self) -> OrchidConfigSnapshot | None:
        """Re-read the config from disk if any file changed.

        Returns the new snapshot, or ``None`` when nothing changed.
        """
        if not self.has_changes():
            return None
        return self._reload()


# ──────────────────────────────────────────────────────────────────
# MD Config Watcher
# ──────────────────────────────────────────────────────────────────


class OrchidConfigWatcher(OrchidConfigWatcherBase):
    """Change detector for Markdown-based configuration.

    Tracks ``orchid.md`` and every ``.md`` in the agents directory by
    their SHA-256 hashes.  Call :meth:`has_changes` on demand (e.g. at
    the start of an API request) to detect edits / additions / deletions
    without a background thread.
    """

    def __init__(
        self,
        *,
        root_path: str | Path,
        agents_dir: str | Path,
        initial_config: OrchidAgentsConfig,
        initial_hashes: dict[str, str],
    ) -> None:
        self._root_path = Path(root_path).resolve()
        self._agents_dir = Path(agents_dir).resolve()
        super().__init__(
            initial_config=initial_config,
            initial_hashes=initial_hashes,
            root_path=self._root_path,
        )

    def _current_hashes(self) -> dict[str, str]:
        """Recompute SHA-256 for every tracked file plus any new ``.md``
        entries in the agents directory.

        Returns an empty dict when the root file has been deleted (the
        watcher signals a change so the caller can decide what to do).
        """
        hashes: dict[str, str] = {}

        # Root config
        if self._root_path.exists():
            hashes[str(self._root_path)] = compute_sha256(self._root_path.read_bytes())
        else:
            return {}

        # Agent files — known paths
        for path_str in self._snapshot.file_hashes:
            if path_str == str(self._root_path):
                continue
            p = Path(path_str)
            if p.exists():
                hashes[path_str] = compute_sha256(p.read_bytes())

        # Agent files — new entries not yet tracked
        if self._agents_dir.exists() and self._agents_dir.is_dir():
            for md_path in sorted(self._agents_dir.glob("*.md")):
                key = str(md_path.resolve())
                if key not in hashes:
                    hashes[key] = compute_sha256(md_path.read_bytes())

        return hashes

    def _reload(self) -> OrchidConfigSnapshot:
        """Force a full reload from disk."""
        config, file_hashes = load_md_config(self._root_path, agents_dir=self._agents_dir)
        self._snapshot = OrchidConfigSnapshot(
            config=config,
            file_hashes=file_hashes,
            root_path=self._root_path,
        )
        logger.info(
            "[ConfigWatcher] Reloaded %d agent(s) from %s",
            len(config.agents),
            self._root_path.name,
        )
        return self._snapshot

    def _reload_single_agent(self, name: str) -> OrchidConfigSnapshot | None:
        """Reload a single agent's ``.md`` file and merge it into the
        current config.

        Returns a new snapshot when the file exists and its hash
        differs; ``None`` when the file is unchanged or missing.
        """
        agent_path = self._agents_dir / f"{name}.md"
        if not agent_path.exists():
            return None

        new_hash = compute_sha256(agent_path.read_bytes())
        old_hash = self._snapshot.file_hashes.get(str(agent_path.resolve()))
        if old_hash == new_hash:
            return None

        agent_md = load_markdown_file(agent_path)
        agent_data = _merge_agent_md(agent_md)

        new_config = self._snapshot.config.model_copy(deep=True)
        if name not in new_config.agents:
            new_config.agents[name] = agent_data  # type: ignore[assignment]
        else:
            existing = new_config.agents[name]
            for key, value in agent_data.items():
                setattr(existing, key, value)

        new_hashes = dict(self._snapshot.file_hashes)
        new_hashes[str(agent_path.resolve())] = new_hash

        self._snapshot = OrchidConfigSnapshot(
            config=new_config,
            file_hashes=new_hashes,
            root_path=self._root_path,
        )
        logger.info("[ConfigWatcher] Reloaded agent '%s'", name)
        return self._snapshot

    # ── Public aliases ─────────────────────────────────────

    def reload(self) -> OrchidConfigSnapshot:
        return self._reload()

    def reload_agent(self, name: str) -> OrchidConfigSnapshot | None:
        return self._reload_single_agent(name)


# ──────────────────────────────────────────────────────────────────
# YAML Config Watcher (feature parity)
# ──────────────────────────────────────────────────────────────────


class OrchidYamlConfigWatcher(OrchidConfigWatcherBase):
    """Change detector for YAML-based configuration.

    Tracks ``orchid.yml`` and ``agents.yaml`` by SHA-256 hash.
    Same interface as :class:`OrchidConfigWatcher` for uniform
    consumer code.
    """

    def __init__(
        self,
        *,
        orchid_yml_path: str | Path,
        agents_yaml_path: str | Path,
        initial_config: OrchidAgentsConfig,
    ) -> None:
        self._orchid_yml = Path(orchid_yml_path).resolve()
        self._agents_yaml = Path(agents_yaml_path).resolve()
        initial_hashes: dict[str, str] = {}
        if self._orchid_yml.exists():
            initial_hashes[str(self._orchid_yml)] = compute_sha256(self._orchid_yml.read_bytes())
        if self._agents_yaml.exists():
            initial_hashes[str(self._agents_yaml)] = compute_sha256(self._agents_yaml.read_bytes())
        super().__init__(
            initial_config=initial_config,
            initial_hashes=initial_hashes,
            root_path=self._agents_yaml,
        )

    def _current_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path_str in self._snapshot.file_hashes:
            p = Path(path_str)
            if p.exists():
                hashes[path_str] = compute_sha256(p.read_bytes())
        return hashes

    def _reload(self) -> OrchidConfigSnapshot:
        config = _load_yaml_config(self._agents_yaml)
        new_hashes: dict[str, str] = {}
        if self._orchid_yml.exists():
            new_hashes[str(self._orchid_yml)] = compute_sha256(self._orchid_yml.read_bytes())
        if self._agents_yaml.exists():
            new_hashes[str(self._agents_yaml)] = compute_sha256(self._agents_yaml.read_bytes())
        self._snapshot = OrchidConfigSnapshot(
            config=config,
            file_hashes=new_hashes,
            root_path=self._agents_yaml,
        )
        logger.info("[YamlWatcher] Reloaded %d agent(s)", len(config.agents))
        return self._snapshot

    def _reload_single_agent(self, name: str) -> OrchidConfigSnapshot | None:
        """Re-read ``agents.yaml`` and return a new snapshot if an agent
        named *name* changed.

        Because YAML config lives in a single file, this is equivalent
        to :meth:`reload` — we re-read the entire file and check if
        the target agent differs.
        """
        if not self._agents_yaml.exists():
            return None

        new_hash = compute_sha256(self._agents_yaml.read_bytes())
        old_hash = self._snapshot.file_hashes.get(str(self._agents_yaml))
        if old_hash == new_hash:
            return None

        new_config = _load_yaml_config(self._agents_yaml)
        if name not in new_config.agents:
            return None

        old_agent = self._snapshot.config.agents.get(name)
        new_agent = new_config.agents[name]
        if old_agent is not None and old_agent == new_agent:
            return None

        return self._reload()

    # ── Public aliases ─────────────────────────────────────

    def reload(self) -> OrchidConfigSnapshot:
        return self._reload()

    def reload_agent(self, name: str) -> OrchidConfigSnapshot | None:
        return self._reload_single_agent(name)


# ── Backward-compat aliases ──────────────────────────────────

ConfigSnapshot = OrchidConfigSnapshot
YamlConfigWatcher = OrchidYamlConfigWatcher
