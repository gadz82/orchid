"""YAML schema for the Pollen + Bloom events block.

The whole block is opt-in: when ``events:`` is omitted (or
``events.enabled`` is ``False``) the framework boots exactly as
without it — no producers started, no processors started, no DB
writes against the events tables.  All Pydantic models forbid
unknown keys so typos and removed fields surface as clear validation
errors instead of silent drift.

Three identity-claim subclasses form a discriminated union:

- ``service_account`` — the platform acts under a named service
  identity (e.g. a ``digest-bot`` token).
- ``addressed_to_user`` — the platform acts under a service identity
  but tags the resulting auth context with a ``user_id`` extracted
  from the signal (used for user-scoped RAG without impersonation).
- ``act_as_user`` — full user impersonation.  The processor calls
  ``OrchidIdentityResolver.mint_for_user(tenant_key, user_id)``;
  triggers using this mode are validated at boot to ensure the
  configured resolver actually implements ``mint_for_user``.

Pluggable components (queue, scheduler, store, producers, processors,
middleware, validators) are all referenced by dotted import path.  The
loader resolves them via ``orchid_ai.config.registry.get_class`` at
boot — same pattern as the rest of the framework.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Identity claim — discriminated union ─────────────────────


class _IdentityClaimBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceAccountIdentity(_IdentityClaimBase):
    """Run as a named platform service identity."""

    mode: Literal["service_account"] = "service_account"
    name: str


class AddressedToUserIdentity(_IdentityClaimBase):
    """Run as a service identity but tagged with a user id extracted
    from the signal payload.  Used when the platform acts on a user's
    behalf without impersonating them (e.g. building a digest scoped to
    that user's data)."""

    mode: Literal["addressed_to_user"] = "addressed_to_user"
    service_account: str
    # JMESPath into the signal envelope (e.g. "signal.user_id" or
    # "payload.actor.id").  Resolved by the processor at fire time.
    user_id_from: str


class ActAsUserIdentity(_IdentityClaimBase):
    """Full user impersonation — the processor mints a fresh
    ``OrchidAuthContext`` via ``OrchidIdentityResolver.mint_for_user``."""

    mode: Literal["act_as_user"] = "act_as_user"
    user_id_from: str


IdentityClaim = Annotated[
    Union[ServiceAccountIdentity, AddressedToUserIdentity, ActAsUserIdentity],
    Field(discriminator="mode"),
]


# ── Trigger ──────────────────────────────────────────────────


class OrchidTriggerMatchConfig(BaseModel):
    """Match block on a trigger.

    ``signal: cron`` is reserved for time-driven triggers — the
    scheduler emits synthetic ``cron`` signals on the dispatcher's
    normal path.  ``cron:`` is required when ``signal == "cron"`` and
    rejected otherwise; the validator below enforces the pairing.
    """

    signal: str
    cron: str | None = None
    when: str | None = None  # JMESPath boolean expression

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _cron_only_with_cron_signal(self) -> "OrchidTriggerMatchConfig":
        if self.cron is not None and self.signal != "cron":
            raise ValueError("trigger.on.cron is only valid when trigger.on.signal == 'cron'")
        if self.signal == "cron" and self.cron is None:
            raise ValueError("trigger.on.cron is required when trigger.on.signal == 'cron'")
        return self


VisibilityLevel = Literal["actor", "addressed", "tenant", "admin"]


# ── Chat binding (§25) ──────────────────────────────────────


class ChatBinding(BaseModel):
    """Opt-in binding from a signal to a real user chat (§25).

    A signal MAY carry one of these.  When the matched trigger has
    ``respect_chat_binding=true`` and the resolved auth has write
    permission on the target chat, the resulting Bloom's final
    ``AIMessage`` lands in that chat with ``metadata.origin="bloom"``.
    Without all three conditions the binding is ignored (§25.2):
    integrators can't smuggle messages into user chats by hand-
    crafting a binding.

    ``source_message_id`` (§LS5) anchors the in-chat live progress
    card under the user message that produced the binding.  Only
    set automatically when an agent calls ``emit_signal(chat_id="self")``
    inside a real chat turn — for cross-chat emissions the field
    defaults to ``None`` so the frontend renders in the bottom
    fallback dock instead of mis-anchoring under another chat's
    message id.
    """

    chat_id: str
    mode: Literal["append_final_message", "append_with_metadata"] = "append_final_message"
    on_failure: Literal["post_error", "silent"] = "post_error"
    source_message_id: str | None = None

    model_config = ConfigDict(extra="forbid")


# ── Trigger emit config ─────────────────────────────────────


# Allowed combinations of (identity flavour, visibility) per §26.3.
_VISIBILITY_MATRIX: dict[str, set[VisibilityLevel]] = {
    "act_as_user": {"actor", "admin"},
    "addressed_to_user": {"addressed", "tenant", "admin"},
    "service_account": {"tenant", "admin"},
}


class OrchidTriggerEmitConfig(BaseModel):
    agent: str
    prompt_template: str
    identity: IdentityClaim
    # §25 — opt-in.  ``respect_chat_binding=true`` plus
    # ``identity.mode=service_account`` is rejected below: a pure
    # service account has no user-of-record, so writing to "the user's
    # chat" is undefined.
    respect_chat_binding: bool = False
    # §26 — explicit visibility override.  ``None`` means the registry
    # computes the default from ``identity.mode`` at boot:
    #   act_as_user        → actor
    #   addressed_to_user  → addressed
    #   service_account    → admin
    visibility: VisibilityLevel | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _binding_requires_user_identity(self) -> "OrchidTriggerEmitConfig":
        # Privilege-escalation guard, §25.3 rule 1.  Same check is
        # repeated at trigger-registry time so a hand-crafted YAML
        # bypassing Pydantic still fails at boot.
        if self.respect_chat_binding and isinstance(self.identity, ServiceAccountIdentity):
            raise ValueError(
                "respect_chat_binding=true requires identity.mode of "
                "'act_as_user' or 'addressed_to_user' (see spec §25.3)"
            )
        return self

    @model_validator(mode="after")
    def _visibility_matches_identity(self) -> "OrchidTriggerEmitConfig":
        # §26.3 — reject forbidden (identity, visibility) combos at
        # config load.  Default visibility (None) is always allowed;
        # the registry computes it.
        if self.visibility is None:
            return self
        identity_mode = self.identity.mode  # discriminator literal
        allowed = _VISIBILITY_MATRIX.get(identity_mode, set())
        if self.visibility not in allowed:
            raise ValueError(
                f"visibility {self.visibility!r} is not allowed with "
                f"identity.mode {identity_mode!r}; allowed values are "
                f"{sorted(allowed)} (see spec §26.3)"
            )
        return self


class OrchidTriggerRetryConfig(BaseModel):
    """Per-trigger retry policy for the supervisor invocation.  Distinct
    from queue retry, which is governed by the queue config."""

    max: int = Field(default=0, ge=0)
    backoff: Literal["fixed", "linear", "exponential"] = "exponential"
    jitter: bool = True
    initial_delay_seconds: float = Field(default=1.0, gt=0.0)
    max_delay_seconds: float = Field(default=300.0, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _max_ge_initial(self) -> "OrchidTriggerRetryConfig":
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("trigger.retry.max_delay_seconds must be >= initial_delay_seconds")
        return self


class OrchidTriggerConfig(BaseModel):
    id: str
    on: OrchidTriggerMatchConfig
    emits: OrchidTriggerEmitConfig
    retry: OrchidTriggerRetryConfig = Field(default_factory=OrchidTriggerRetryConfig)
    parallelism: Literal["per_user", "per_tenant", "unbounded"] = "per_user"

    model_config = ConfigDict(extra="forbid")


# ── Schedule ─────────────────────────────────────────────────


class OrchidScheduleConfig(BaseModel):
    """A schedule emits one synthetic ``cron`` signal per fire — the
    same ingest path everything else uses."""

    id: str
    cron: str | None = None
    interval_seconds: int | None = Field(default=None, gt=0)
    trigger_id: str
    identity: IdentityClaim
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _exactly_one_timing(self) -> "OrchidScheduleConfig":
        cron_set = self.cron is not None
        interval_set = self.interval_seconds is not None
        if cron_set == interval_set:
            raise ValueError("schedule must specify exactly one of cron / interval_seconds")
        return self


# ── Ingestion source registry ────────────────────────────────


class OrchidValidatorConfig(BaseModel):
    """Per-source validator (HMAC, bearer, mTLS, …)."""

    class_path: str = Field(alias="class")
    secret_ref: str | None = None
    extra_args: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OrchidIngestionSourceConfig(BaseModel):
    id: str
    validator: OrchidValidatorConfig
    allowed_types: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class OrchidEventsIngestionConfig(BaseModel):
    """Webhook source registry block.  Renamed from the spec's
    ``ingestion`` to avoid collision with the rag-side ingestion
    config that lives at the same import root."""

    sources: list[OrchidIngestionSourceConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# ── Pluggable component refs ─────────────────────────────────
#
# Most events components are referenced by a dotted import path plus
# free-form extra_args.  Two component refs (queue, processor) carry
# enough common knobs that we model them explicitly so YAML autocomplete
# / validation catches typos in the hot fields.


class _ComponentRef(BaseModel):
    """Generic 'instantiate the class at this dotted path' shape."""

    class_path: str = Field(alias="class")
    extra_args: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _ComponentRefAllowExtra(BaseModel):
    """Used for components whose YAML carries inline knobs alongside
    ``class:`` — the subclasses below pin the well-known knobs and
    leave the rest free."""

    class_path: str = Field(alias="class")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OrchidQueueConfig(_ComponentRefAllowExtra):
    notify_enabled: bool = True
    poll_interval_ms: int = Field(default=200, ge=10)
    lease_seconds: int = Field(default=30, ge=1)
    max_attempts: int = Field(default=5, ge=1)
    dead_letter_table: str = "signal_queue_dead_letter"


class OrchidProcessorConfig(_ComponentRefAllowExtra):
    concurrency: int = Field(default=4, ge=1)
    poll_interval_ms: int = Field(default=200, ge=10)
    lease_seconds: int = Field(default=30, ge=1)
    max_attempts: int = Field(default=5, ge=1)
    drain_timeout_seconds: float = Field(default=10.0, gt=0.0)


# ── Top-level events block ───────────────────────────────────


class OrchidEventsConfig(BaseModel):
    """Root events configuration.

    When :attr:`enabled` is ``False`` (the default) the rest of the
    block is still parsed for early validation, but no producers /
    processors are started and the dispatcher / store / queue are not
    constructed.  This keeps the 'opt-out is zero overhead' contract.
    """

    enabled: bool = False

    store: _ComponentRef | None = None
    queue: OrchidQueueConfig | None = None
    scheduler: _ComponentRef | None = None

    producers: list[_ComponentRef] = Field(default_factory=list)
    processors: list[OrchidProcessorConfig] = Field(default_factory=list)
    middleware: list[_ComponentRef] = Field(default_factory=list)

    ingestion: OrchidEventsIngestionConfig = Field(default_factory=OrchidEventsIngestionConfig)
    schedules: list[OrchidScheduleConfig] = Field(default_factory=list)
    triggers: list[OrchidTriggerConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _required_when_enabled(self) -> "OrchidEventsConfig":
        if not self.enabled:
            return self
        missing: list[str] = []
        if self.store is None:
            missing.append("events.store")
        if self.queue is None:
            missing.append("events.queue")
        if not self.processors:
            missing.append("events.processors")
        if missing:
            raise ValueError("events.enabled is true but required keys are missing: " + ", ".join(missing))

        # Cross-field check: every schedule must point at a real trigger
        # in this config (forward references aren't supported).
        trigger_ids = {t.id for t in self.triggers}
        for sched in self.schedules:
            if sched.trigger_id not in trigger_ids:
                raise ValueError(f"schedule {sched.id!r} references unknown trigger {sched.trigger_id!r}")
        # And: schedule emits 'cron' signals, so the matching trigger
        # must declare ``signal: cron``.  Catches the easy mistake of
        # wiring a schedule to a webhook trigger.
        triggers_by_id = {t.id: t for t in self.triggers}
        for sched in self.schedules:
            tr = triggers_by_id[sched.trigger_id]
            if tr.on.signal != "cron":
                raise ValueError(
                    f"schedule {sched.id!r} targets trigger {tr.id!r} "
                    f"but that trigger is bound to signal "
                    f"{tr.on.signal!r}, not 'cron'"
                )
        return self


__all__ = [
    "ActAsUserIdentity",
    "AddressedToUserIdentity",
    "ChatBinding",
    "IdentityClaim",
    "OrchidEventsConfig",
    "OrchidEventsIngestionConfig",
    "OrchidIngestionSourceConfig",
    "OrchidProcessorConfig",
    "OrchidQueueConfig",
    "OrchidScheduleConfig",
    "OrchidTriggerConfig",
    "OrchidTriggerEmitConfig",
    "OrchidTriggerMatchConfig",
    "OrchidTriggerRetryConfig",
    "OrchidValidatorConfig",
    "ServiceAccountIdentity",
    "VisibilityLevel",
]
