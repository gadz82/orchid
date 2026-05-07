"""In-memory trigger registry — built from YAML at boot.

Two responsibilities:

1. **Compile** each ``OrchidTriggerConfig`` into an ``_OrchidYamlTrigger``
   instance — a thin :class:`OrchidTrigger` whose ``matches`` /
   ``build_job_spec`` are pure functions of the signal.
2. **Validate at registration time**.  This is the moment Pollen +
   Bloom catches misconfiguration: an ``act_as_user`` trigger pointed at
   a resolver that can't mint identities should fail at boot, not at
   first-fire-time.  Phase 1 ships with the agent-existence check and
   the JMESPath parseability check; the resolver probe (which runs the
   real ``mint_for_user`` call) lands in the phase that adds
   ``mint_for_user`` to the resolver ABC, but the registry already
   exposes the hook.

Match order: linear scan over the configured triggers, JMESPath
applied per ``when:``.  This is fine for trigger sets up to ~hundreds;
larger sets should plug in an alternative ``TriggerRegistry``
implementation that indexes by ``signal.type`` first.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable, Protocol

import jmespath

from ..config.schema_events import (
    ActAsUserIdentity,
    AddressedToUserIdentity,
    OrchidTriggerConfig,
    ServiceAccountIdentity,
    VisibilityLevel,
)
from ..core.events.errors import (
    MintingProbeUnsupportedError,
    OrchidIdentityNotMintableError,
    TriggerMatchError,
    TriggerRegistrationError,
)
from ..core.events.job import JobSpec, RetryPolicy
from ..core.events.signal import Signal
from ..core.events.trigger import OrchidTrigger, TriggerRegistry

# §26.3 — same matrix the Pydantic validator uses; mirrored here for
# defence-in-depth at registration time.
_VISIBILITY_MATRIX: dict[str, frozenset[VisibilityLevel]] = {
    "act_as_user": frozenset({"actor", "admin"}),
    "addressed_to_user": frozenset({"addressed", "tenant", "admin"}),
    "service_account": frozenset({"tenant", "admin"}),
}

# §26.2 — default visibility per identity flavour.
_DEFAULT_VISIBILITY: dict[str, VisibilityLevel] = {
    "act_as_user": "actor",
    "addressed_to_user": "addressed",
    "service_account": "admin",
}

_logger = logging.getLogger(__name__)


class _ResolverProbe(Protocol):
    """Minimal protocol the registry needs from the identity resolver
    at registration time.  Phase 1 keeps this loose so we don't have to
    import the full :class:`OrchidIdentityResolver` here — the
    ``mint_for_user`` extension method lands in the phase that wires
    the real probe."""

    def mint_for_user(self, tenant_key: str, user_id: str) -> Any: ...


class _OrchidYamlTrigger(OrchidTrigger):
    """Compiled form of a single :class:`OrchidTriggerConfig`."""

    def __init__(
        self,
        config: OrchidTriggerConfig,
        *,
        when_compiled: jmespath.parser.ParsedResult | None,
        visibility: VisibilityLevel,
    ) -> None:
        self._config = config
        self._when = when_compiled
        self._visibility = visibility

    @property
    def visibility(self) -> VisibilityLevel:
        """Resolved visibility (config override OR identity-based default)."""
        return self._visibility

    @property
    def respect_chat_binding(self) -> bool:
        return self._config.emits.respect_chat_binding

    @property
    def trigger_id(self) -> str:
        return self._config.id

    @property
    def parallelism(self) -> str:
        return self._config.parallelism

    @property
    def retry_policy(self) -> RetryPolicy:
        r = self._config.retry
        return RetryPolicy(
            max_attempts=r.max,
            backoff=r.backoff,
            jitter=r.jitter,
            initial_delay_seconds=r.initial_delay_seconds,
            max_delay_seconds=r.max_delay_seconds,
        )

    @property
    def identity_claim(self) -> dict[str, Any]:
        return _identity_claim_to_dict(self._config.emits.identity)

    def matches(self, signal: Signal) -> bool:
        if signal.type != self._config.on.signal:
            return False
        if self._when is None:
            return True
        try:
            return bool(self._when.search(signal.as_match_dict()))
        except Exception as exc:  # JMESPath edge case (e.g. missing field)
            raise TriggerMatchError(
                f"trigger {self.trigger_id!r} when-expression raised on signal {signal.signal_id}: {exc!r}"
            ) from exc

    def build_job_spec(self, signal: Signal) -> JobSpec:
        identity = self.identity_claim
        prompt = _render_prompt(
            self._config.emits.prompt_template,
            signal=signal,
            identity=identity,
        )
        # §26 — visibility / visibility_user_id are immutable from this
        # point on.  ``actor`` and ``addressed`` carry a user id; the
        # other levels do not (the migration's CHECK constraint enforces
        # the same shape at the DB layer).
        visibility = self._visibility
        visibility_user_id: str | None = None
        if visibility in ("actor", "addressed"):
            visibility_user_id = resolve_user_id_for_signal(identity, signal=signal) or signal.user_id
        # §25 — chat_binding only flows into the JobSpec when the
        # trigger opts in.  Without ``respect_chat_binding=true`` we
        # drop the binding here so the runner never sees it.
        chat_binding: dict[str, Any] | None = None
        if self.respect_chat_binding and signal.chat_binding is not None:
            chat_binding = dict(signal.chat_binding)
        return JobSpec(
            trigger_id=self.trigger_id,
            signal_id=signal.signal_id,
            agent_name=self._config.emits.agent,
            prompt=prompt,
            identity_claim=identity,
            correlation_id=signal.correlation_id,
            parallelism_key=_compute_parallelism_key(self._config.parallelism, signal=signal, identity=identity),
            visibility=visibility,
            visibility_user_id=visibility_user_id,
            chat_binding=chat_binding,
        )


class InMemoryTriggerRegistry(TriggerRegistry):
    """Default registry used by the asyncio processor."""

    def __init__(self) -> None:
        self._by_id: dict[str, OrchidTrigger] = {}

    def register(self, trigger: OrchidTrigger) -> None:
        if trigger.trigger_id in self._by_id:
            raise TriggerRegistrationError(f"trigger {trigger.trigger_id!r} is already registered")
        self._by_id[trigger.trigger_id] = trigger

    def get(self, trigger_id: str) -> OrchidTrigger | None:
        return self._by_id.get(trigger_id)

    def all(self) -> Iterable[OrchidTrigger]:
        return list(self._by_id.values())

    def find_matches(self, signal: Signal) -> Iterable[OrchidTrigger]:
        # Linear scan — an early ``signal.type`` filter is the cheapest
        # short-circuit.  Larger trigger sets should plug in their own
        # ``TriggerRegistry`` keyed by type.
        out: list[OrchidTrigger] = []
        for trigger in self._by_id.values():
            try:
                if trigger.matches(signal):
                    out.append(trigger)
            except TriggerMatchError:
                # Bad ``when:`` against a malformed payload — log but
                # don't crash the processor.  The signal is still
                # ack'able after the other triggers are evaluated.
                _logger.exception(
                    "trigger %s when-expression failed on signal %s",
                    trigger.trigger_id,
                    signal.signal_id,
                )
        return out


def build_registry_from_config(
    triggers: list[OrchidTriggerConfig],
    *,
    known_agents: set[str],
    identity_resolver: _ResolverProbe | None = None,
) -> InMemoryTriggerRegistry:
    """Compile + validate + register a list of trigger configs.

    Validation order — fail fast on the cheapest checks first:

    1. Trigger ID uniqueness.
    2. ``emits.agent`` must exist in ``known_agents``.
    3. ``when:`` must parse as JMESPath (parse-time only — runtime
       errors against malformed payloads are caught later).
    4. ``act_as_user`` triggers probe ``mint_for_user``: resolvers that
       can't mint at all (raise :class:`MintingProbeUnsupportedError`)
       are rejected here, naming both the trigger ID and the resolver
       class.  Resolvers that raise the regular
       :class:`OrchidIdentityNotMintableError` for the probe user pass
       — that just means 'this specific user has no token', which is
       fine at boot.

    ``identity_resolver`` may be ``None`` in tests / phase-1 callers
    that don't yet wire a resolver; in that case the probe is skipped
    with a debug log (registration still fails on the other checks).
    """
    registry = InMemoryTriggerRegistry()
    seen: set[str] = set()
    for cfg in triggers:
        if cfg.id in seen:
            raise TriggerRegistrationError(f"duplicate trigger id: {cfg.id!r}")
        seen.add(cfg.id)

        if cfg.emits.agent not in known_agents:
            raise TriggerRegistrationError(
                f"trigger {cfg.id!r} targets unknown agent "
                f"{cfg.emits.agent!r} (known agents: "
                f"{sorted(known_agents) or 'none'})"
            )

        when_compiled: jmespath.parser.ParsedResult | None = None
        if cfg.on.when is not None:
            try:
                when_compiled = jmespath.compile(cfg.on.when)
            except jmespath.exceptions.ParseError as exc:
                raise TriggerRegistrationError(
                    f"trigger {cfg.id!r} has unparseable when-expression {cfg.on.when!r}: {exc}"
                ) from exc

        if isinstance(cfg.emits.identity, ActAsUserIdentity):
            _probe_minting(
                trigger_id=cfg.id,
                resolver=identity_resolver,
            )

        # §25.3 rule 1 — defence-in-depth on top of the Pydantic validator.
        if cfg.emits.respect_chat_binding and isinstance(cfg.emits.identity, ServiceAccountIdentity):
            raise TriggerRegistrationError(
                f"trigger {cfg.id!r} declares respect_chat_binding=true with "
                f"identity.mode='service_account' — service accounts have no "
                f"user-of-record, so chat binding is forbidden (see spec §25.3)"
            )

        # §26.3 — defence-in-depth on top of the Pydantic visibility
        # validator.  ``None`` defers to the identity-flavour default
        # below.
        visibility = _resolve_visibility(cfg)

        trigger = _OrchidYamlTrigger(cfg, when_compiled=when_compiled, visibility=visibility)
        registry.register(trigger)

    return registry


def _resolve_visibility(cfg: OrchidTriggerConfig) -> VisibilityLevel:
    """Compute the trigger's resolved visibility (override OR default).

    Re-validates the (identity, visibility) matrix even though the
    Pydantic layer already does so — registries built directly from
    dataclasses (tests / programmatic flows) skip Pydantic, and the
    registration-time check is the last guard before YAML state lands
    in the runtime registry.
    """
    identity_mode = cfg.emits.identity.mode
    explicit = cfg.emits.visibility
    if explicit is None:
        return _DEFAULT_VISIBILITY[identity_mode]
    allowed = _VISIBILITY_MATRIX.get(identity_mode, frozenset())
    if explicit not in allowed:
        raise TriggerRegistrationError(
            f"trigger {cfg.id!r} declares visibility={explicit!r} with "
            f"identity.mode={identity_mode!r}; allowed values are "
            f"{sorted(allowed)} (see spec §26.3)"
        )
    return explicit


# ── Helpers ─────────────────────────────────────────────────


def _probe_minting(*, trigger_id: str, resolver: _ResolverProbe | None) -> None:
    """Probe the resolver's ``mint_for_user`` capability.

    A resolver that raises :class:`MintingProbeUnsupportedError` does
    not implement minting *at all* — that is fatal for any
    ``act_as_user`` trigger and surfaces as a
    :class:`TriggerRegistrationError`.  A plain
    :class:`OrchidIdentityNotMintableError` just signals 'this probe
    user has no credentials', which is fine at boot.

    The probe is sync from the registry's perspective even when
    ``mint_for_user`` is itself async — Phase 3 made the resolver
    method async on the ABC, so we drive the coroutine via
    :mod:`asyncio` here to keep registration ergonomically blocking
    (consumers call ``build_registry_from_config`` from sync boot
    code; making the registry async would ripple through every
    integrator).
    """
    if resolver is None:
        _logger.debug(
            "skipping mint probe for trigger %r — no resolver supplied",
            trigger_id,
        )
        return
    try:
        result = resolver.mint_for_user("__probe__", "__probe__")
        if hasattr(result, "__await__"):
            # Async resolver — drive the coroutine to its raise point.
            import asyncio as _asyncio

            try:
                _asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — safe to use ``asyncio.run``.
                _asyncio.run(result)
            else:
                # Inside an event loop (rare during registration but
                # possible in tests / orchid-cli's ``serve``).  Drive
                # the coroutine in a fresh loop on a worker thread to
                # avoid deadlocking the caller's loop.
                import concurrent.futures as _futures

                with _futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_asyncio.run, result)
                    future.result()
    except MintingProbeUnsupportedError as exc:
        raise TriggerRegistrationError(
            f"trigger {trigger_id!r} requires act_as_user but resolver "
            f"{exc.resolver_class!r} does not implement mint_for_user"
        ) from exc
    except OrchidIdentityNotMintableError:
        # Expected — the probe user is a sentinel, the resolver said
        # 'I can mint, just not for this user'.  Trigger is OK.
        return
    except Exception as exc:
        # Something unrelated blew up — surface it but don't pretend
        # the probe succeeded.
        raise TriggerRegistrationError(f"trigger {trigger_id!r} mint_for_user probe raised {exc!r}") from exc


def _identity_claim_to_dict(
    claim: ServiceAccountIdentity | AddressedToUserIdentity | ActAsUserIdentity,
) -> dict[str, Any]:
    # ``model_dump`` would also work but we want a plain dict that
    # ``core/events`` can carry without depending on Pydantic.
    if isinstance(claim, ServiceAccountIdentity):
        return {"mode": "service_account", "name": claim.name}
    if isinstance(claim, AddressedToUserIdentity):
        return {
            "mode": "addressed_to_user",
            "service_account": claim.service_account,
            "user_id_from": claim.user_id_from,
        }
    if isinstance(claim, ActAsUserIdentity):
        return {"mode": "act_as_user", "user_id_from": claim.user_id_from}
    raise TypeError(f"unknown identity claim type: {type(claim).__name__}")


def _render_prompt(template: str, *, signal: Signal, identity: dict[str, Any]) -> str:
    """Tiny ``{{var}}`` placeholder substitution.

    Supports a handful of well-known names that are convenient for the
    YAML author: ``{{tenant_key}}``, ``{{user_id}}``,
    ``{{signal.type}}``, ``{{signal.source}}``, ``{{payload.<key>}}``
    (top-level only).  Returns the template verbatim when no
    substitutions match — broken or unrecognised placeholders surface
    naturally as visible ``{{junk}}`` in the rendered prompt rather
    than crashing the trigger."""
    out = template
    replacements: dict[str, str] = {
        "tenant_key": signal.tenant_key,
        "user_id": signal.user_id or "",
        "signal.id": str(signal.signal_id),
        "signal.type": signal.type,
        "signal.source": signal.source,
        "signal.user_id": signal.user_id or "",
        "signal.tenant_key": signal.tenant_key,
        "identity.mode": str(identity.get("mode", "")),
    }
    for key, value in replacements.items():
        out = out.replace("{{" + key + "}}", value)
    # payload.<key>
    payload = signal.payload or {}
    for key, value in payload.items():
        token = "{{payload." + str(key) + "}}"
        if token in out:
            out = out.replace(token, str(value))
    return out


def _compute_parallelism_key(mode: str, *, signal: Signal, identity: dict[str, Any]) -> str:
    if mode == "unbounded":
        # Caller-side note: the processor still needs *something* to
        # key on so that retries of the same job don't race.  Use the
        # signal id which is unique by definition.
        return f"unbounded:{signal.signal_id}"
    if mode == "per_tenant":
        return f"tenant:{signal.tenant_key}"
    # per_user (default)
    if identity.get("mode") == "service_account":
        return f"sa:{signal.tenant_key}:{identity.get('name', 'unknown')}"
    user_id = signal.user_id or "anonymous"
    return f"user:{signal.tenant_key}:{user_id}"


def resolve_user_id_for_signal(identity: dict[str, Any], *, signal: Signal) -> str | None:
    """Extract the ``user_id`` for an ``act_as_user`` /
    ``addressed_to_user`` claim out of the signal envelope, using the
    JMESPath in ``user_id_from``.

    Returns ``None`` when:

    - the identity claim is ``service_account`` (no user dimension), OR
    - the JMESPath resolves to nothing.

    Public so the processor can call it without re-implementing the
    JMESPath plumbing.  Lives in this module because the registry is
    where JMESPath is already wired in."""
    mode = identity.get("mode")
    if mode == "service_account":
        return None
    expr = identity.get("user_id_from")
    if not isinstance(expr, str):
        return None
    try:
        result = jmespath.search(expr, signal.as_match_dict())
    except Exception:
        return None
    return str(result) if result is not None else None


# Re-export Callable for type-stub friendly checking — keeps this
# module self-contained for the test that asserts no
# ``orchid_ai.events`` import in core/.
_Callable = Callable
