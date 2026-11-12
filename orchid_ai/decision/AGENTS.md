# decision/ — Decision Providers

## What This Package Is

Built-in implementations of `orchid_ai.core.decision.OrchidDecisionModel` plus the transport,
shared answer validation, and the composition factory. The top-level `orchid_ai` package exports
the ABC and typed data only; adapters live here.

## Files

| File | Responsibility |
|---|---|
| `answers.py` | Presence-aware confidence/probability/score validation, refusals, usage |
| `http.py` | Bounded HTTPS transport: egress checks, resolved-address validation, deadline, breaker, size caps |
| `state_questions.py` | State-and-questions dialect adapter |
| `openai_compatible.py` | OpenAI-compatible decisions dialect adapter |
| `llm.py` | Any LangChain chat model via structured output |
| `stub.py` | Deterministic provider for tests and the offline example |
| `factory.py` | Protocol registry, API-key resolution, state-transform wrapping |
| `registry.py` | Decision-point specs, built-ins, validators |
| `observability.py` | The single structured `orchid.decision` line |

## Rules

- The transport is the only outbound path. Do not add provider SDK imports; the boundary test
  rejects them.
- Answers never compose tool arguments. Names only.
- Every provider failure is `OrchidDecisionError` with a stable code; never leak state, bodies, or
  credentials.
- `required=True` questions make a missing answer an `invalid_response`; callers fall back.
- New protocols: `register_decision_protocol(name, builder)` and validate the provider shape in
  `config/schema_decision.py`.

## Enabling a New Decision Point

A point is declarative plumbing, not a provider feature. Adding one never changes an existing
configuration: the point stays inert until YAML selects a provider for it.

1. **Register the spec** before `load_config()` runs (for example in an import-time hook of the
   consuming package). `register_decision_point` rejects duplicate names and can never replace a
   built-in.

   ```python
   from orchid_ai.core.decision import OrchidDecisionFallbackPolicy, OrchidDecisionPointSpec
   from orchid_ai.decision.registry import register_decision_point

   register_decision_point(
       OrchidDecisionPointSpec(
           name="my_point",
           scope="agent",
           fallback=OrchidDecisionFallbackPolicy(
               mode="configured_behaviour",
               description="Keep the behaviour configured on the agent.",
           ),
           description="One sentence the operator can read.",
           validate_config=_validate_my_point_options,
       )
   )
   ```

2. **Validate YAML in the point's validator.** It receives the raw, explicitly-set binding
   (`enabled`, `model`, `confidence_threshold`, `options`, …) and must raise `ValueError` with an
   actionable message. The config model only enforces a mapping with non-empty string keys, so
   check JSON-safe values and point-specific semantics here. Configuration validation runs for
   both the global block and every per-agent override, so a bad option fails before a request
   starts.

3. **Choose the scope deliberately.**
   - `fleet`: resolved once before an agent is known. Per-agent overrides are rejected.
   - `agent`: resolvable globally and overridden under `agents.<name>.decisions.points.<point>`;
     `enabled: false` disables an inherited global binding.
   - Declare `max_candidates` only if the adapter really bounds the candidate set. YAML may narrow
     the ceiling but can never raise it; a spec without a ceiling rejects `max_candidates`.

4. **Implement the adapter against the binding protocol.** Consume
   `OrchidDecisionPointBinding`: `spec`, `provider`, `point`, `confidence_threshold`,
   `max_candidates`, `options`, `decide`, `metadata`. The declared fallback lives on
   `binding.spec.fallback.mode` (the concrete `DecisionBinding` also exposes a `fallback_policy`
   convenience). Never import a concrete provider, transport, or config model. Resolve with
   `bind_decision_points(config, runtime)` for fleet/global points or
   `resolve_decision_binding(config, runtime, point, agent_name=...)` per agent. The factory builds
   only providers referenced by a configured point or by a legacy `agents.<name>.decision_model`;
   an unreferenced declaration is never constructed.

5. **Own the fallback in the adapter.** The spec's fallback mode is documentation plus a test
   contract; the framework does not execute it. On a missing/refused/low-confidence answer,
   timeout, provider error, unsafe candidate, or budget exhaustion, return the pre-existing
   behaviour (configured strategy/model/tools, no delegation, round cap, …). Never raise out of
   the seam.

6. **Test the seam at every level.** Unit-test the spec, the YAML validator, and the resolution chain
   (global → per-agent → default). Add a network-free integration fixture under
   `tests/decision/fixtures/` and drive it through `load_config` → `build_decision_models` →
   `bind_decision_points`/`build_graph`; cover an accepted decision and one fallback reason.
   Pin that legacy points still resolve to `None` and that the same YAML without the new block
   builds nothing. `tests/decision/conftest.py` guards against live `httpx.AsyncClient`
   construction. The integration test is white-box: it builds the graph but reaches the pipeline
   and binding through private `_rag_pipeline`/`_retrieval_strategy_binding` attributes, so it is
   not a full `Orchid` execution. Adapter behaviour is covered by the focused tests in
   `test_retrieval_strategy.py` (and the other point suites); no run-through-graph test is needed.

7. **Emit one `log_decision(...)` per evaluation.** Pass only `point`, `path` (`decision` or the
   legacy `llm_fallback` token), `provider`, `latency_ms`, `confidence`, `outcome`, and
   `fallback_reason`; attach `binding.metadata()` to graph state when useful. Never log state,
   payload, prompts, candidate text, or credentials. `test_new_points_integration.py` is the
   reference for the fixture + asserted line shape.

## Known Follow-ups (P-004 … P-009)

- **Guardrails:** F1a — `fm_agent.SecretDetectionGuardrail` has no `fail_action` constructor; the
  deterministic builder skips it on `TypeError`. Deterministic rail `config:` flattening ignores
  options such as `max_length.max_characters`. HITL/checkpoint: per-agent input redaction happens
  in the agent-node update, so an interrupt before it can retain the raw message. Deeper nested
  agents, mini-agent forks, and peer skill calls can bypass per-agent rails. ADR-002/Q-001 need an
  addendum: deterministic rails do not use `on_failure`; safety/PII is opt-in via `type: decision`
  with an explicit policy.
- **Composition:** bootstrap can emit a duplicate missing-provider warning for one composition
  error.
- **Tool choice:** the point is wired through `MCPDispatcher.fetch`, not the default agentic path;
  docs/examples still describe `llm_decides` as an LLM prompt. `stop_reason` is log-only;
  mini-agent loops bypass `agentic_loop`; two tool-selection adapters remain to consolidate.
- **Memory:** the point is per-seam, not per-turn; `synthesizer.py` and `sequential_advancer.py`
  retain legacy writes; `keep` is not a turn veto; `persist` can repeat across seams. Bounded
  user/assistant and summary excerpts are sent unredacted. Agent `summarize` cannot seed the first
  summary; only the route seam or `persist` can. Accepted `persist` is logged before the write and
  storage failures are swallowed. Nested child agents/sub-supervisors now receive root summary
  config/memory (intentional M6 alignment); confirm the cross-agent behaviour is desired.
- **Events:** `orchid-api`/`orchid-cli` do not pass `decision_binding` to `start_events`; wire
  `Orchid.runtime.get_decision_binding(events_cfg.decision.point)`. Dedup is provider-owned;
  `cron` signals are gated; there is no explicit decision timeout; hot reload does not refresh the
  processor binding; event type/source are untrusted prompt input; gated decisions are not
  persisted; first delivery is `attempt == 1` by queue convention.
- **Ingestion:** `orchid-api`/`orchid-cli` still call `documents.pipeline` directly; wire
  `agent.extract_text`/`agent.ingest_document` or pass the binding. `Orchid.agents` is typed
  `dict[str, OrchidAgent]`, so a Protocol/cast or ABC methods are needed. Allowlist validation for
  `options.parsers`/`.strategies` is runtime-only; add a registry validator. A confident
  cross-format parser choice is trusted; two provider calls per document; the first 500 chars and
  the filename are sent unredacted; a provider-selected strategy uses its own chunk defaults; the
  structured line does not name parser vs strategy phase; whitespace-heavy prefixes can yield an
  empty snapshot.
- **Delegation:** mini-agent forks run before `GenericAgent.run()` and bypass the gate; peer-agent
  skill chains can reach a peer's external tools; accepted dispatch logs raw tool args in the
  pre-existing `AgenticLoop`; fail-closed/empty-catalog paths emit no structured line;
  `SkillDetector._last` can race under concurrency; `options.agents` is runtime-validated only;
  delegation is not charged to the per-turn budget; up to 4,000 chars of raw query reach the
  provider; custom agents overriding `run()` remain outside the gate by design.
