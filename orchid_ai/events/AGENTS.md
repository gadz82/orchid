# events/ — Pollen + Bloom (event-driven activation layer)

## What this package is

Concrete implementations of the events ABCs declared in
``orchid_ai/core/events/``. The naming convention:

- **Pollen** is the signal substrate — ingest, persistence, queue.
  Producers normalise external events into ``SignalEnvelope``s and
  hand them to the dispatcher; the dispatcher persists + enqueues in
  one transaction; the queue is the durable buffer.
- **Bloom** is the execution layer — processors drain the queue,
  triggers match signals to ``JobSpec``s, and the ``JobRunner``
  invokes the existing LangGraph supervisor under a synthesised
  ``OrchidAuthContext``. A ``JobRun`` is the unit of execution.

The whole layer is **opt-in**. With ``events.enabled: false`` (default
when the block is omitted) nothing in this package runs and no
storage rows are written.

## Boundary rules (MUST follow)

1. **``orchid_ai/core/events/`` has zero external dependencies.** Only
   stdlib + ``langchain_core`` (in line with the rest of ``core/``).
   No ``pydantic`` imports inside ``core/events/``; Pydantic models
   live next door in ``orchid_ai/config/schema_events.py``.
2. **Import direction is one-way.** Code under
   ``orchid_ai/events/`` may import from
   ``orchid_ai/core/events/``. The reverse is forbidden — verified by
   ``tests/test_dependency_boundaries.py``.
3. **The dispatcher does NOT match triggers.** ``OrchidSignalDispatcher``
   only persists and enqueues. Trigger matching, identity resolution,
   and supervisor invocation all run in the processor on the dequeue
   side of the queue. Anything that pulls match logic back into the
   dispatcher breaks the latency contract.
4. **Identity resolution lives in the processor.** The signal carries
   an *identity claim* (a structured dict, validated against the
   discriminated union in ``schema_events``); the processor turns the
   claim into an ``OrchidAuthContext`` via
   ``OrchidIdentityResolver``. No identity work happens on the
   ingest path.
5. **Internal emissions go through ``dispatcher.ingest``.** No
   in-process fast-path that bypasses persistence. ``OrchidAgent
   .emit_signal`` (added when the agent-binding lands) and
   ``DispatcherSignalEmitter`` both forward to the same ingest method.
6. **Idempotency by construction.** ``UNIQUE (source, dedupe_key)``
   on signals; ``UNIQUE (trigger_id, signal_id, attempt_number)`` on
   job runs. The in-memory backend enforces both via dicts; the
   Postgres / SQLite backends will use real constraints. Retries
   become new ``JobRun`` rows — never in-place updates.
7. **No vendor / product names** — same hard rule as the rest of
   ``orchid/``. Use ``acme.example``, ``MyAuthExchange``,
   ``INTEGRATOR_*`` placeholders. Platform-specific work belongs in
   consumer projects.

## Layout

```
orchid_ai/events/
  __init__.py                   facade module — keeps top-level surface tiny
  registry.py                   InMemoryTriggerRegistry + JMESPath compile/match
  queues/
    __init__.py
    inmemory.py                 InMemorySignalQueue + InMemoryJobStore +
                                InMemorySignalStore + InMemoryScheduleStore +
                                InMemoryTriggerStore (single test-friendly bundle)
    sqlite.py                   SQLiteSignalQueue (durable, single-process)
    postgres.py                 PostgresSignalQueue (FOR UPDATE SKIP LOCKED,
                                pg_notify on commit)
    relay.py                    RelayingSignalQueue + BusPublisher ABC +
                                InMemoryBusPublisher (for tests / demos)
  backends/
    __init__.py
    sqlite.py                   SQLiteEventStorage facade + four narrow stores
                                (signals / jobs / schedules / triggers)
    postgres.py                 PostgresEventStorage facade + four narrow stores
  schedulers/
    __init__.py
    apscheduler.py              APSchedulerBackend (no SQLAlchemy — durability
                                lives in the schedules table, jobs are
                                re-registered on every boot)
  producers/
    __init__.py
    internal.py                 DispatcherSignalEmitter
    scheduler.py                SchedulerProducer (cron / interval → cron signals)
    http.py                     HTTPIngestionProducer (lazy FastAPI router)
    relay_recovery.py           RelayRecoveryProducer (publish-then-mark sweep)
  auth/
    __init__.py
    base.py                     SignalAuthValidator ABC + SignalAuthRequest
    hmac.py                     HMACValidator (constant-time SHA-256)
    bearer.py                   BearerValidator
  streaming.py                  BloomEvent + ChatBloomEvent +
                                BloomEventStream (channel-keyed pub/sub)
  visibility.py                 Run-visibility SQL fragment + Python predicate
  processors/
    __init__.py
    asyncio_pool.py             AsyncioWorkerPoolProcessor
  runners/
    __init__.py
    graph_runner.py             GraphJobRunner (chat binding +
                                final-message persistence + on_failure)
  AGENTS.md                     this file
  CLAUDE.md                     symlink → AGENTS.md
```

A sibling package outside ``events/`` carries the identity-mixin
helper (kept off ``core/`` because the mixin's optional
:class:`OrchidTokenRefresher` collaborator pulls in HTTP):

```
orchid_ai/identity/
  __init__.py                   re-exports OAuthMintingMixin + OrchidTokenRefresher
  oauth_minting.py              the mixin + Protocol
```

## Components

What this package ships:

- All ABCs in ``orchid_ai/core/events/`` (``Signal``, ``Trigger``,
  ``JobSpec``, ``JobRun``, ``OrchidSignalQueue``,
  ``OrchidSignalProducer``, ``OrchidSignalProcessor``,
  ``OrchidJobRunner``, ``OrchidSignalEmitter``,
  ``OrchidSignalStore``, ``OrchidJobStore``, ``OrchidScheduleStore``,
  ``OrchidTriggerStore``, ``SignalIngestMiddleware``,
  ``OrchidSignalDispatcher`` plus the exception hierarchy).
- ``InMemorySignalQueue`` + the four in-memory stores under one roof.
- ``SQLiteSignalQueue`` + ``SQLiteEventStorage`` (durable single-process
  queue, FOREIGN-KEY-cascading job runs, transactional outbox).
- ``PostgresSignalQueue`` + ``PostgresEventStorage`` (``FOR UPDATE
  SKIP LOCKED`` dequeue, optional ``pg_notify`` on commit, atomic
  outbox via shared connection / pool).
- ``RelayingSignalQueue`` — publish-then-mark adapter for external
  buses with the ``BusPublisher`` ABC.  ``RelayRecoveryProducer``
  drives the durable ``relay_status`` column through a periodic
  sweep that re-publishes pending rows.
- ``InMemoryTriggerRegistry`` with JMESPath ``when:`` compilation and
  registration-time validation hooks (agent existence, JMESPath
  parseability, mint-probe stub).
- ``DispatcherSignalEmitter``.
- ``APSchedulerBackend`` wrapping ``apscheduler.AsyncIOScheduler`` —
  no SQLAlchemy dependency.  The ``schedules`` table is the
  durability boundary; APScheduler's in-memory jobstore is
  re-populated on every boot.
- ``SchedulerProducer`` driving APScheduler against an
  ``OrchidScheduleStore``: registers cron / interval jobs, fires
  synthetic ``cron`` signals through ``dispatcher.ingest`` with
  ``dedupe_key`` = ``"<schedule_id>:<fire_iso>"``, and updates
  ``last_fire_at`` / ``next_fire_at`` after each fire.
- ``AsyncioWorkerPoolProcessor`` — both a long-running ``start`` /
  ``stop`` mode and a synchronous ``process_until_idle`` mode for
  unit tests.
- Pydantic models in ``orchid_ai/config/schema_events.py`` plus the
  ``events: OrchidEventsConfig | None = None`` field on
  ``OrchidAgentsConfig``.
- Migration ``v001_initial_schema`` carries every framework-owned
  table — chat, MCP outbound, MCP inbound gateway, AND the seven
  events tables (``signals``, ``signal_queue``,
  ``signal_queue_dead_letter``, ``triggers``, ``schedules``,
  ``job_runs``, ``signal_sources``).  Single root migration.
- ``OrchidIdentityResolver.resolve_service_account`` and
  ``mint_for_user`` — concrete-with-raising-defaults on the ABC.
  The mint probe at registration drives the async resolver from sync
  boot code by spinning a worker-thread loop when the caller is
  already inside one.
- ``OAuthMintingMixin`` in ``orchid_ai/identity/`` — composable mixin
  for resolvers whose only consumer-facing MCP servers are
  ``oauth``-mode.  Reads :class:`OrchidMCPTokenStore`, optionally
  refreshes via an injected :class:`OrchidTokenRefresher` Protocol,
  and produces :class:`OrchidAuthContext` instances ready for the
  processor.
- ``OrchidAuthContext.roles: frozenset[str]`` — concrete-with-default
  field on the existing core type.  Reserved value ``"admin"`` for
  the run-visibility filter; round-trips through
  :meth:`to_storage_dict` / :meth:`from_storage_dict`.
- ``ChatBinding`` Pydantic model + ``OrchidTriggerEmitConfig
  .respect_chat_binding`` opt-in flag.  Pydantic-level rejection
  of ``respect_chat_binding=true`` + ``service_account``; registry-
  level rejection of the same combo as defence-in-depth.
- ``OrchidTriggerEmitConfig.visibility: VisibilityLevel | None``
  field with an (identity, visibility) compatibility matrix
  enforced at both Pydantic and registry layers.  Defaults computed
  per identity flavour (``act_as_user → actor``,
  ``addressed_to_user → addressed``, ``service_account → admin``).
- ``JobSpec`` carries ``visibility`` + ``visibility_user_id`` + the
  resolved ``chat_binding`` (only when the matching trigger opts in),
  and the migration's CHECK constraint enforces
  ``visibility_user_id`` shape per visibility level on insert.
- ``GraphJobRunner``: runs an injected graph invoker, calls
  ``OrchidSessionWarmer.warm_for_user(auth)`` via the processor's
  pre-run hook, resolves chat binding via ``_resolve_chat_binding``,
  persists the final ``AIMessage`` to :class:`OrchidChatStorage` with
  ``metadata.origin="bloom"`` on success, posts a brief failure
  message when ``binding.on_failure="post_error"``.  Cross-user
  smuggling attempts are rejected at runtime regardless of what the
  signal carried — the runner re-validates ownership through the
  resolved auth.
- ``OrchidAgent.emit_signal`` — async method on the base class.
  Inherits ``tenant_key`` / ``user_id`` from the current
  ``OrchidAuthContext`` set by the graph wrapper.
  ``chat_id="self"`` binds to the current chat; explicit
  ``chat_id`` is re-validated by the runner.
- ``OrchidChatStorage.get_chat_metadata`` and ``can_write`` defaults
  — concrete on the ABC so existing implementations keep working
  unchanged.
- ``SignalAuthValidator`` ABC + built-in ``HMACValidator`` and
  ``BearerValidator``; constant-time comparisons; HMAC validates
  against the raw body so payloads can be parsed safely AFTER the
  signature check.
- ``HTTPIngestionProducer`` — lazy-imports FastAPI so the library
  stays platform-agnostic, mounts at the configured ``mount`` path,
  returns 202 + ``{signal_id, deduplicated}`` on success.  Honours
  ``X-Orchid-Source`` / ``X-Orchid-Signature`` / ``Idempotency-Key``
  headers and a JSON body that maps onto ``SignalEnvelope``.
- ``RelayRecoveryProducer`` — periodic sweep over
  ``signals WHERE relay_status='pending_publish'``; flips
  ``published`` after a successful re-publish; leaves rows pending
  on publisher failure for the next tick.  Plays nicely with
  :class:`RelayingSignalQueue`'s publish-then-mark contract.
  In-memory + SQLite + Postgres stores all expose
  ``list_by_relay_status`` so the sweep doesn't scan the whole
  table.
- ``BloomEventStream`` — in-process channel-keyed pub/sub.  The
  generic primitives are ``subscribe(channel)`` /
  ``publish(channel, event)``; per-run callers use the
  ``subscribe_run`` / ``publish_run`` thin wrappers that delegate
  to channel ``f"run:{run_id}"``.  Subscribers get an async
  iterator that closes after a terminal event or after an idle
  timeout.  Slow subscribers evict their oldest event rather than
  block the publisher.  Wired into
  :class:`AsyncioWorkerPoolProcessor` — ``bloom.run.queued`` /
  ``.started`` / ``.finished`` are emitted automatically once a
  stream is injected.
- ``visibility.build_run_filter_clause`` (Postgres + SQLite SQL
  fragments) and ``visibility.run_is_visible`` (in-memory predicate)
  — used by the orchid-api routers to enforce run visibility on
  every ``SELECT FROM job_runs`` and the analogous ``signals``
  query.  Cross-tenant access is always rejected, even for admins.

## In-chat streaming

Live in-chat progress for chat-bound Blooms.  See
[`chat-binding.md`](../../../.knowledge/documentation/concepts/chat-binding.md)
§"Live progress" for the user-facing model.  Components:

- ``ChatBinding.source_message_id: str | None`` — anchors the
  in-chat progress card under the user message that produced the
  binding.  Auto-populated by
  ``OrchidAgent.emit_signal(chat_id="self")`` from
  ``_current_message_id``; cross-chat emissions default to
  ``None`` (anchoring under another chat's message id is
  meaningless).
- ``OrchidAgent._current_message_id`` plumbed via
  ``orchid_ai/graph/graph.py`` — the agent wrapper sets it from
  the latest ``HumanMessage`` and restores the prior value in a
  ``finally`` block, so concurrent invocations don't leak per-call
  context across each other.
- ``BloomEventStream`` channel API exposes ``subscribe(channel)``
  / ``publish(channel, event)``.  The pre-existing per-run callers
  use the ``subscribe_run`` / ``publish_run`` wrappers that
  delegate to ``f"run:{run_id}"``.  Two well-known channel
  families: ``run:{run_id}`` (operator stream) and ``chat:{chat_id}``
  (in-chat progress).
- ``ChatBloomEvent`` + ``_wrap_for_chat`` in ``streaming.py`` —
  three event types: ``chat.bloom.attached`` (collapses
  ``bloom.run.queued`` + ``bloom.run.started``; consumer dedupes
  by run id), ``chat.bloom.tick`` (forwards a redacted payload —
  only ``kind`` / ``agent`` / ``tool`` / ``status`` / ``message``,
  no raw tool result bodies), ``chat.bloom.finished``
  (carries ``status`` / ``finished_at`` / ``error`` but NOT the
  run ``result`` — the final AIMessage flows through chat reload).
- Processor dual-publish branch in
  ``processors/asyncio_pool.py:_publish_event(event, *, run=None)``
  — the run channel and chat channel each have an independent
  ``try/except`` so a chat-channel publish failure cannot affect
  the run channel publish.  Non-bound runs publish to the run
  channel only.
- ``OrchidJobStore.list(chat_binding_chat_id=…, statuses=[…])``
  filter for the chat-events endpoint's discovery query —
  implemented for in-memory, SQLite (``json_extract``), and
  Postgres (JSONB ``->'chat_binding'->>'chat_id'``).
- New endpoint ``GET /chats/{chat_id}/events/stream`` in
  ``orchid_api/routers/chat_events.py`` with discovery →
  subscribe pattern.  ``require_chat_owner_or_admin`` returns
  404 (never 403); cross-tenant always 404.

## YAML surface (additive, fully opt-in)

```yaml
events:
  enabled: true

  store: { class: orchid_ai.events.backends.postgres.PostgresEventStore }
  queue: { class: orchid_ai.events.queues.postgres.PostgresSignalQueue }
  scheduler: { class: orchid_ai.events.schedulers.apscheduler.APSchedulerBackend }

  producers:
    - class: orchid_ai.events.producers.http.HTTPIngestionProducer
      mount: /signals
    - class: orchid_ai.events.producers.scheduler.SchedulerProducer
    - class: orchid_ai.events.producers.internal.InternalEmissionProducer

  processors:
    - class: orchid_ai.events.processors.asyncio_pool.AsyncioWorkerPoolProcessor
      concurrency: 4

  ingestion:
    sources:
      - id: support-system
        validator:
          class: orchid_ai.events.auth.HMACValidator
          secret_ref: env:SUPPORT_HMAC_SECRET
        allowed_types: [support.ticket.created]

  schedules:
    - id: morning-digest-cron
      cron: "0 7 * * 1-5"
      trigger_id: morning-digest
      identity: { mode: service_account, name: digest-bot }

  triggers:
    - id: morning-digest
      on: { signal: cron, cron: "0 7 * * 1-5" }
      emits:
        agent: notifications
        prompt_template: "Build the morning digest for {{tenant_key}}"
        identity: { mode: service_account, name: digest-bot }
      retry: { max: 3, backoff: exponential, jitter: true }
      parallelism: unbounded
```

Notes:

- Every Pydantic model under ``schema_events`` uses
  ``model_config = {"extra": "forbid"}`` — typos surface immediately.
- ``schedule.cron`` and ``schedule.interval_seconds`` are exclusive.
  The validator rejects 'both' and 'neither'.
- A schedule's ``trigger_id`` must exist in this same file AND must
  point at a trigger whose ``on.signal == "cron"``.
- ``identity.mode`` is the discriminator. ``act_as_user`` triggers
  are probed at boot — a resolver that can't mint at all (raises
  ``MintingProbeUnsupportedError``) gets a deterministic boot-time
  failure naming both the trigger and the resolver class.

## Common pitfalls

- **Reaching for the dispatcher from inside a producer.** Don't.
  Producers receive the dispatcher in ``start(dispatcher)`` and
  should hold their own reference rather than walking back through
  any global.
- **Implementing trigger matching with a coroutine.** ``matches()``
  is sync and pure on purpose — no I/O. Anything that needs an
  async lookup belongs in the runner, not in the trigger.
- **Trying to atomically update a ``JobRun`` instead of inserting a
  new attempt.** The unique constraint
  ``(trigger_id, signal_id, attempt_number)`` is what gives Bloom
  its replay safety; a retry MUST insert a new row with
  ``attempt_number + 1``.
- **Forgetting the per-key parallelism lock.** The asyncio worker
  pool serialises by ``parallelism_key`` to avoid two parallel
  Blooms racing on the same user's MCP cache state. Skipping this
  in a custom processor reintroduces races.
- **Mutating ``Signal`` instances.** Signals are frozen dataclasses;
  the only legitimate post-insert mutation is flipping
  ``relay_status`` for the external-bus relay path, and that goes
  through ``OrchidSignalStore.update_relay_status`` which rebuilds
  the dataclass.
