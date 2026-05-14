"""Tests for on-demand config watchers — SHA-256 change detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchid_ai.config.md_loader import load_md_config
from orchid_ai.config.watcher import ConfigSnapshot, OrchidConfigWatcher, YamlConfigWatcher


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _write_orchid_md(root: Path, extra: str = "") -> None:
    root.write_text(
        "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\n" + extra + "---\n",
        encoding="utf-8",
    )


def _write_agent_md(agents_dir: Path, name: str, description: str, prompt: str = "Prompt.") -> None:
    (agents_dir / f"{name}.md").write_text(f"---\ndescription: {description}\n---\n\n{prompt}", encoding="utf-8")


def _make_md_snapshot(tmp_path: Path) -> tuple[OrchidConfigWatcher, Path, Path]:
    root = tmp_path / "orchid.md"
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write_orchid_md(root)
    _write_agent_md(agents_dir, "agent_a", "Agent A", "Prompt A.")
    config, hashes = load_md_config(root, agents_dir=agents_dir)
    watcher = OrchidConfigWatcher(
        root_path=root,
        agents_dir=agents_dir,
        initial_config=config,
        initial_hashes=hashes,
    )
    return watcher, root, agents_dir


# ──────────────────────────────────────────────────────────────────
# ConfigSnapshot
# ──────────────────────────────────────────────────────────────────


class TestConfigSnapshot:
    def test_snapshot_is_frozen(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_orchid_md(root)
        config, hashes = load_md_config(root, agents_dir=agents_dir)
        snap = ConfigSnapshot(config=config, file_hashes=hashes, root_path=root.resolve())
        with pytest.raises(Exception):
            snap.file_hashes = {}  # type: ignore[misc]

    def test_snapshot_stores_hashes(self, tmp_path):
        root = tmp_path / "orchid.md"
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_orchid_md(root)
        config, hashes = load_md_config(root, agents_dir=agents_dir)
        snap = ConfigSnapshot(config=config, file_hashes=hashes, root_path=root.resolve())
        assert str(root.resolve()) in snap.file_hashes
        assert len(snap.file_hashes[str(root.resolve())]) == 64


# ──────────────────────────────────────────────────────────────────
# OrchidConfigWatcher
# ──────────────────────────────────────────────────────────────────


class TestOrchidConfigWatcherHasChanges:
    def test_no_changes_when_files_unchanged(self, tmp_path):
        watcher, _, _ = _make_md_snapshot(tmp_path)
        assert watcher.has_changes() is False

    def test_detects_root_change(self, tmp_path):
        watcher, root, _ = _make_md_snapshot(tmp_path)
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\nchanged: true\n---\n",
            encoding="utf-8",
        )
        assert watcher.has_changes() is True

    def test_detects_agent_change(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        (agents_dir / "agent_a.md").write_text("---\ndescription: Changed\n---\n\nNew prompt.", encoding="utf-8")
        assert watcher.has_changes() is True

    def test_detects_new_agent_file(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        _write_agent_md(agents_dir, "agent_b", "Agent B")
        assert watcher.has_changes() is True

    def test_detects_deleted_agent_file(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        (agents_dir / "agent_a.md").unlink()
        assert watcher.has_changes() is True

    def test_detects_deleted_root(self, tmp_path):
        watcher, root, _ = _make_md_snapshot(tmp_path)
        root.unlink()
        assert watcher.has_changes() is True


class TestOrchidConfigWatcherChangedFiles:
    def test_empty_when_no_changes(self, tmp_path):
        watcher, _, _ = _make_md_snapshot(tmp_path)
        assert watcher.changed_files() == []

    def test_returns_changed_agent_file(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        agent_path = agents_dir / "agent_a.md"
        agent_path.write_text("---\ndescription: Updated\n---\n\nUpdated.", encoding="utf-8")
        changed = watcher.changed_files()
        assert str(agent_path.resolve()) in changed

    def test_new_file_listed(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        _write_agent_md(agents_dir, "agent_c", "Agent C")
        changed = watcher.changed_files()
        new_path = str((agents_dir / "agent_c.md").resolve())
        assert new_path in changed


class TestOrchidConfigWatcherReload:
    def test_reload_returns_new_snapshot(self, tmp_path):
        watcher, root, agents_dir = _make_md_snapshot(tmp_path)
        snap_before = watcher.snapshot
        new_snap = watcher.reload()
        assert new_snap is not snap_before
        assert new_snap.config.agents.keys() == snap_before.config.agents.keys()

    def test_reload_picks_up_changes(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        _write_agent_md(agents_dir, "agent_b", "New Agent B", "New prompt.")
        new_snap = watcher.reload()
        assert "agent_b" in new_snap.config.agents
        assert new_snap.config.agents["agent_b"].description == "New Agent B"

    def test_reload_if_changed_when_changed(self, tmp_path):
        watcher, root, _ = _make_md_snapshot(tmp_path)
        root.write_text(
            "---\nversion: '1'\ndefaults:\n  rag:\n    enabled: false\nsupervisor:\n  assistant_name: Changed\n---\n",
            encoding="utf-8",
        )
        result = watcher.reload_if_changed()
        assert result is not None

    def test_reload_if_changed_when_unchanged(self, tmp_path):
        watcher, _, _ = _make_md_snapshot(tmp_path)
        result = watcher.reload_if_changed()
        assert result is None


class TestOrchidConfigWatcherReloadAgent:
    def test_no_change_returns_none(self, tmp_path):
        watcher, _, _ = _make_md_snapshot(tmp_path)
        result = watcher.reload_agent("agent_a")
        assert result is None

    def test_change_returns_new_snapshot(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        agent_path = agents_dir / "agent_a.md"
        agent_path.write_text(
            "---\ndescription: Updated Agent A\n---\n\nNew prompt text.",
            encoding="utf-8",
        )
        result = watcher.reload_agent("agent_a")
        assert result is not None
        assert result.config.agents["agent_a"].description == "Updated Agent A"
        assert result.config.agents["agent_a"].prompt == "New prompt text."

    def test_missing_file_returns_none(self, tmp_path):
        watcher, _, _ = _make_md_snapshot(tmp_path)
        result = watcher.reload_agent("nonexistent")
        assert result is None

    def test_has_changes_false_after_reload(self, tmp_path):
        watcher, _, agents_dir = _make_md_snapshot(tmp_path)
        (agents_dir / "agent_a.md").write_text("---\ndescription: Reloaded\n---\n\nReloaded prompt.", encoding="utf-8")
        result = watcher.reload_agent("agent_a")
        assert result is not None
        assert watcher.has_changes() is False


# ──────────────────────────────────────────────────────────────────
# YamlConfigWatcher
# ──────────────────────────────────────────────────────────────────


class TestYamlConfigWatcher:
    def test_initial_snapshot(self, tmp_path):
        import yaml

        from orchid_ai.config.loader import load_config

        agents_yml = tmp_path / "agents.yaml"
        agents_yml.write_text(
            yaml.dump(
                {
                    "version": "1",
                    "defaults": {"rag": {"enabled": False}},
                    "agents": {"agent_a": {"description": "Test", "prompt": "Prompt."}},
                }
            ),
            encoding="utf-8",
        )
        orchid_yml = tmp_path / "orchid.yml"
        orchid_yml.write_text("", encoding="utf-8")

        config = load_config(agents_yml)
        watcher = YamlConfigWatcher(
            orchid_yml_path=orchid_yml,
            agents_yaml_path=agents_yml,
            initial_config=config,
        )
        assert watcher.has_changes() is False
        assert "agent_a" in watcher.snapshot.config.agents

    def test_detects_change(self, tmp_path):
        import yaml

        from orchid_ai.config.loader import load_config

        agents_yml = tmp_path / "agents.yaml"
        data = {
            "version": "1",
            "defaults": {"rag": {"enabled": False}},
            "agents": {"agent_a": {"description": "Test", "prompt": "Prompt."}},
        }
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        orchid_yml = tmp_path / "orchid.yml"
        orchid_yml.write_text("", encoding="utf-8")

        config = load_config(agents_yml)
        watcher = YamlConfigWatcher(
            orchid_yml_path=orchid_yml,
            agents_yaml_path=agents_yml,
            initial_config=config,
        )
        assert watcher.has_changes() is False

        data["agents"]["agent_a"]["description"] = "Changed"
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        assert watcher.has_changes() is True

    def test_reload_if_changed(self, tmp_path):
        import yaml

        from orchid_ai.config.loader import load_config

        agents_yml = tmp_path / "agents.yaml"
        data = {
            "version": "1",
            "defaults": {"rag": {"enabled": False}},
            "agents": {"agent_a": {"description": "Test", "prompt": "Prompt."}},
        }
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        orchid_yml = tmp_path / "orchid.yml"
        orchid_yml.write_text("", encoding="utf-8")

        config = load_config(agents_yml)
        watcher = YamlConfigWatcher(
            orchid_yml_path=orchid_yml,
            agents_yaml_path=agents_yml,
            initial_config=config,
        )

        data["agents"]["agent_a"]["description"] = "Updated"
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        snap = watcher.reload_if_changed()
        assert snap is not None
        assert snap.config.agents["agent_a"].description == "Updated"

    def test_reload_agent(self, tmp_path):
        import yaml

        from orchid_ai.config.loader import load_config

        agents_yml = tmp_path / "agents.yaml"
        data = {
            "version": "1",
            "defaults": {"rag": {"enabled": False}},
            "agents": {"agent_a": {"description": "Test", "prompt": "Prompt."}},
        }
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        orchid_yml = tmp_path / "orchid.yml"
        orchid_yml.write_text("", encoding="utf-8")

        config = load_config(agents_yml)
        watcher = YamlConfigWatcher(
            orchid_yml_path=orchid_yml,
            agents_yaml_path=agents_yml,
            initial_config=config,
        )

        data["agents"]["agent_a"]["prompt"] = "New prompt."
        agents_yml.write_text(yaml.dump(data), encoding="utf-8")
        snap = watcher.reload_agent("agent_a")
        assert snap is not None
        assert snap.config.agents["agent_a"].prompt == "New prompt."

    def test_changed_files(self, tmp_path):
        import yaml

        from orchid_ai.config.loader import load_config

        agents_yml = tmp_path / "agents.yaml"
        agents_yml.write_text(
            yaml.dump(
                {
                    "version": "1",
                    "defaults": {"rag": {"enabled": False}},
                    "agents": {"agent_a": {"description": "Test", "prompt": "Prompt."}},
                }
            ),
            encoding="utf-8",
        )
        orchid_yml = tmp_path / "orchid.yml"
        orchid_yml.write_text("", encoding="utf-8")

        config = load_config(agents_yml)
        watcher = YamlConfigWatcher(
            orchid_yml_path=orchid_yml,
            agents_yaml_path=agents_yml,
            initial_config=config,
        )

        assert watcher.changed_files() == []
        agents_yml.write_text(
            yaml.dump(
                {
                    "version": "1",
                    "defaults": {"rag": {"enabled": False}},
                    "agents": {"agent_a": {"description": "Changed", "prompt": "Prompt."}},
                }
            ),
            encoding="utf-8",
        )
        changed = watcher.changed_files()
        assert str(agents_yml.resolve()) in changed
