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

## Rules

- The transport is the only outbound path. Do not add provider SDK imports; the boundary test
  rejects them.
- Answers never compose tool arguments. Names only.
- Every provider failure is `OrchidDecisionError` with a stable code; never leak state, bodies, or
  credentials.
- `required=True` questions make a missing answer an `invalid_response`; callers fall back.
- New protocols: `register_decision_protocol(name, builder)` and validate the provider shape in
  `config/schema_decision.py`.
