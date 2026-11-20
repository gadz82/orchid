# Extending Data Pipelines

Data pipelines are open for extension at four points: deterministic functions,
custom step classes, new step kinds, and class-based shapes. All four are
operator surfaces — the code you reference runs with the pipeline's authority —
so the trust boundary is described at the end.

- [Deterministic functions](#deterministic-functions)
- [Custom step classes](#custom-step-classes)
- [Registering a new job kind](#registering-a-new-job-kind)
- [Class-based shapes](#class-based-shapes)
- [Custom operators and the trust boundary](#custom-operators-and-the-trust-boundary)
- [Testing an extension](#testing-an-extension)

## Deterministic functions

The lowest-ceremony extension is a function referenced by `uses`:

```python
# my_pkg/steps.py
def score_account(ctx) -> dict:
    account = ctx.graph.get("account:1")
    return {
        "score": 42 if account is not None else 0,
        "customer_id": ctx.input["customer_id"],
    }
```

```yaml
jobs:
  score:
    kind: deterministic
    uses: my_pkg.steps.score_account
    params:
      profile: strict
    writes:
      - id: "score:{{ input.customer_id }}"
        shape: Score
```

The callable receives an `OrchidPipelineStepContext` with:

| Attribute | Contents |
|-----------|----------|
| `run` | `run_id`, pipeline name/version, mode, correlation id, timestamps. |
| `job` | Current job id, kind, attempt, params, map item/index/parent. |
| `input` | The validated run input. |
| `graph` | A read-only `OrchidDataGraphView`. |
| `auth` | The run's `OrchidAuthContext` (tenant, user, tenant-scoped access). |
| `chat_model`, `reader` | Optional runtime model and vector reader. |
| `logger` | A job-scoped logger. |
| `mapping_context` | The full JMESPath context (`input`, `run`, `jobs`, `objects`, `types`, `aliases`, `summary`). |
| `idempotency_key` | Stable key for external side effects. |

Return one of:

- `None` — no output, no writes.
- `dict` — the job output (must be JSON-safe).
- `OrchidPipelineStepResult` — output, requested `OrchidGraphMutation`s, usage
  and metadata.
- a list of `OrchidGraphMutation` — writes with empty output.

A synchronous function runs in a worker thread; an `async def` is awaited on
the event loop. A blocking sync function therefore cannot stall a parallel job.
Raise `OrchidPipelineRetryableError` for transient failures and
`OrchidPipelineStepError` for terminal ones.

## Custom step classes

For reusable logic in a class, subclass `OrchidPipelineStep` and reference the
class from YAML:

```python
# my_pkg/steps.py
from orchid_ai import OrchidPipelineStep, OrchidPipelineStepResult


class ScoreStep(OrchidPipelineStep):
    kind = "custom"

    async def execute(self, ctx) -> OrchidPipelineStepResult:
        score = await score_customer(ctx.input["customer_id"], ctx)
        return OrchidPipelineStepResult(
            output={"score": score},
            usage={"llm_calls": 0},
            metadata={"scorer": "local"},
        )
```

```yaml
jobs:
  score:
    kind: custom
    class: my_pkg.steps.ScoreStep
```

The compiler imports `class` at build time and instantiates it with **no
arguments**, once per custom **job**. Keep the constructor zero-argument and
put dependencies at module level (or resolve them from `ctx`: `auth`,
`chat_model`, `reader`, `call_tool`, `emit_signal`). If you need true dependency
injection, use `register_pipeline_step` and a factory (next section), which
receives the job config and `PipelineDependencies`.

Instances must be **stateless across runs**: all per-run data arrives through
`execute(ctx)`. A step may request graph mutations by returning
`OrchidPipelineStepResult(mutations=(...))`; the session applies them atomically
after the step returns and validates every write against the declared shapes and
relation allowlists.

## Registering a new job kind

Plugins and embedded integrations can register a factory:

```python
from orchid_ai import OrchidPipelineStep, OrchidPipelineStepResult, register_pipeline_step


class ScoreStep(OrchidPipelineStep):
    kind = "score"

    def __init__(self, points: int) -> None:
        self._points = points

    async def execute(self, ctx) -> OrchidPipelineStepResult:
        return OrchidPipelineStepResult(output={"score": self._points})


def _build_score(config, deps):
    return ScoreStep(config.params.get("points", 0))


register_pipeline_step("score", _build_score)
```

`register_pipeline_step(kind, factory, override=False)` rejects duplicate kinds
unless `override=True` (for reloads and tests). In v1 the registry does **not**
protect the built-in kinds: `override=True` can replace any registered factory,
including a built-in. Do not replace a built-in in an application. The factory
receives `(job_config, PipelineDependencies)` and must return an
`OrchidPipelineStep`.

**v1 limitation:** the YAML schema discriminates jobs on the eight built-in
kinds (`deterministic`, `llm`, `agent`, `decision`, `map`, `subpipeline`,
`emit`, `custom`). A registered kind is reachable only from a directly
constructed `CompiledJob`/`CompiledPipeline`; even
`OrchidPipelineDefinitionConfig` validates the eight kinds and rejects
`kind: score`. A plugin that registers a new kind before compilation does not
make `kind: score` valid in YAML. Use `kind: custom` + `class:` for
YAML-declared extension steps, or construct the compiled pipeline in code.

## Class-based shapes

A shape can point at a Pydantic `BaseModel` subclass or a dataclass instead of
declaring fields inline:

```python
# my_pkg/shapes.py
from pydantic import BaseModel, Field


class Score(BaseModel):
    model_config = {"extra": "forbid"}

    value: int = Field(ge=0, le=100)
    band: str
```

```yaml
pipelines:
  shapes:
    Score:
      class: my_pkg.shapes.Score
```

The class is imported at compile time and used through a `TypeAdapter`. It must
be a Pydantic model or a dataclass; anything else fails with a clear shape
error. `version`, `description` and `additional_properties` belong in YAML; the
class owns field validation. Class-based shapes participate in `writes`,
`input`, `output`, LLM structured output and the generated JSON Schema exactly
like field-based shapes.

Use a class-based shape when validation needs custom validators, computed
fields or a type shared with the rest of your application. Prefer inline
`fields` when the shape is example-local.

## Custom operators and the trust boundary

Pipeline extension points are **trusted operator code**, not a sandbox:

- `uses`, `class` and shape `class` are dotted import paths resolved at build
  time. Never build them from user input, request payloads or database rows.
- A custom step runs with the pipeline's `OrchidAuthContext`, model, vector
  reader and tool caller. It can call `ctx.call_tool(...)` and
  `ctx.emit_signal(...)`; treat it with the same review bar as framework code.
- Steps must not place the bearer token or raw credentials in `output`,
  `metadata`, logs or emitted signals. The runtime bounds errors and redacts the
  public result, but it cannot redact a secret you deliberately return.
- Object and relation writes are checked against the declared shapes and the
  job's `links` allowlist after the step returns. A step cannot invent a shape
  or relation type at runtime.
- Keep steps deterministic where possible: same input + graph → same output.
  Use `ctx.idempotency_key` for external side effects so a retry is safe.
- A step must not mutate the graph view; return mutations instead. The session
  owns revision checks, conflict handling and atomic commit.

## Testing an extension

Test the step directly with a small fake context, then compile the pipeline to
exercise the wiring:

```python
async def test_score_step_returns_a_typed_output():
    step = ScoreStep(config, deps)
    result = await step.execute(fake_ctx(input={"customer_id": "C-1"}))
    assert result.output == {"score": 42}
```

For the full path, register the step (or declare `kind: custom`) and run the
pipeline with the fake chat model used by the examples e2e suite. A custom step
that writes objects should assert the graph snapshot (`include_graph=True`) and
the per-job status, not only the final output.

See [`pipelines.md`](./pipelines.md) for the YAML reference, job kinds, result
contract, limits and security notes.
