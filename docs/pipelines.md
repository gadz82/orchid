# Data Pipelines

Orchid data pipelines are typed, run-scoped DAGs declared in `agents.yaml`.
Each job reads from a small information graph, produces a JSON-safe output, and
may write typed objects and relations into the run's graph. The graph lives in
the run session (never in a LangGraph checkpoint), so partial results survive a
failed job without leaking object data into state channels.

The runnable example is [`workspace-py/examples/data_pipeline/`](../../examples/data_pipeline/).
It uses every job kind and a deterministic decision provider; its e2e suite
runs offline with a fake chat model, while the Docker profile uses a local
Ollama model.

- [Concepts](#concepts)
- [Declaring a pipeline](#declaring-a-pipeline)
- [Shapes and relations](#shapes-and-relations)
- [Job kinds](#job-kinds)
- [Mapping context](#mapping-context)
- [Results and interpretation](#results-and-interpretation)
- [Limits](#limits)
- [Facade, CLI and HTTP](#facade-cli-and-http)
- [Security notes](#security-notes)

## Concepts

| Term | Meaning |
|------|---------|
| **Pipeline** | A named DAG of jobs, declared under `pipelines.definitions`. |
| **Job** | One node. Its `kind` selects the step implementation. |
| **Run** | One execution with an immutable `run_id`, validated input and an isolated information graph. |
| **Shape** | A named, versioned schema. Inputs, outputs and graph writes are validated against it. |
| **Object** | A typed node in the information graph (`id`, `type`, `data`). |
| **Relation** | A typed, directed edge between two objects. |
| **Alias** | The latest object written under a name by a job (`aliases.account`). |
| **Interpretation** | The human-readable `summary` returned with the result. |

Pipeline state is deliberately small: the DAG state carries run metadata, job
results and map bookkeeping; the information graph is owned by the run session.
A failed job marks the run `failed` when its `on_error` is `fail` (the default);
`continue` and `skip` leave the run `partial` when other jobs fail.

## Declaring a pipeline

The block lives in the agents configuration:

```yaml
pipelines:
  enabled: true
  defaults:
    model: ollama/llama3.2
    retry:
      max_attempts: 2
  limits:
    max_llm_calls: 20
    max_map_items: 50
  shapes: { ... }
  relations: { ... }
  definitions:
    customer_health:
      description: "Assess a customer and propose an action."
      input:
        shape: Request
      output:
        shape: Report
        from: "jobs.report.output"
      jobs:
        load:
          kind: deterministic
          uses: examples.data_pipeline.steps.load_account
        report:
          kind: llm
          prompt: "Write a report for {{ input.customer_id }}."
          output:
            shape: Report
          depends_on: [load]
      expose:
        as_tool: true
        tool_name: pipeline_customer_health
        description: "Assess a customer and propose a retention action."
      interpretation:
        mode: template
        template: "{{ output.headline }}"
```

`enabled: false` still parses and validates the block (so typos surface early)
but builds no registry, tools or graph. Omitting the block changes nothing.

### Definition fields

| Field | Meaning |
|-------|---------|
| `description` / `version` | Documentation and shape-version hint. |
| `enabled` | Per-definition switch; disabled definitions are not registered. |
| `input` | Exactly one of `shape` or `fields`, plus `defaults`. |
| `output` | A `shape` and the JMESPath `from` that produces it. |
| `jobs` | Job id → job configuration. |
| `expose` | Opt-in agent tool (`as_tool`, `tool_name`, `requires_approval`, `parallel_safe`, `require_role`, `max_tool_result_chars`, `interpretation`). |
| `async` | Bloom launch binding (`enabled`, `signal`/`trigger_id`, `mode`, `timeout_seconds`). |
| `interpretation` | `none`, `template`, `llm` or `caller`; the source of `summary`. |
| `limits` | Per-definition ceilings; values may only tighten the global limits. |

### Job common fields

| Field | Meaning |
|-------|---------|
| `kind` | `deterministic`, `llm`, `agent`, `decision`, `map`, `subpipeline`, `emit` or `custom`. |
| `depends_on` | Job ids this job is scheduled after. See the v1 scheduling note below. |
| `join` | `all` (default) or `any`; use `any` to converge mutually exclusive decision branches. |
| `when` | JMESPath or template gate; a false result records a skip. |
| `writes` | Declarative object writes, validated against a shape. |
| `links` | Declarative relation writes; each type must be declared under `relations`. |
| `retry` | `max_attempts`, `backoff`, `jitter` and delay bounds. |
| `timeout_seconds` | Per-attempt timeout. |
| `on_error` | `fail`, `continue` or `skip`. |
| `metadata` | JSON-safe operator metadata copied into the result. |

## Shapes and relations

Shapes are declared once under `pipelines.shapes` and referenced by name. A
field may declare `type`, `required`, `default`, `enum`, `items`, `properties`,
`additional_properties`, `min`/`max` and `min_length`/`max_length`, `pattern`,
or a `ref` to another shape. Validation happens at the input boundary, at every
write, and at the output projection.

Relations are declared under `pipelines.relations` and typed by their endpoint
shapes:

```yaml
relations:
  has_usage:
    from: [Account]
    to: [Usage]
```

A job writes an object with `writes: [{id, shape, as?, from?}]` and a relation
with `links: [{type, from, to, data?}]`. `from` defaults to the job output;
`as` exposes the latest written object under an alias for later templates.
A job may only write relation types declared in its own `links` (the write
allowlist) and every referenced shape must exist.

## Job kinds

### `deterministic`

Runs trusted operator code resolved from a dotted path. A synchronous callable
runs in a worker thread; a coroutine is awaited directly.

```yaml
load:
  kind: deterministic
  uses: examples.data_pipeline.steps.load_account
  params: {tier: pro}
```

The callable receives a step context (`ctx.input`, `ctx.run`, `ctx.graph`,
`ctx.auth`, `ctx.mapping_context`, `ctx.job`) and returns `None`, a `dict`
(the job output), an `OrchidPipelineStepResult`, or a list of
`OrchidGraphMutation`s.

### `llm`

Renders a prompt from the mapping context and calls the configured chat model.
`output.shape` enables structured output; `rag` optionally retrieves context;
`tools` names built-in tools the model may call. The per-job `model` wins over
the injected runtime model, which wins over `pipelines.defaults.model`.

### `agent`

Runs one top-level leaf agent through the same executor as chat: guardrails,
model routing, MCP skills and inline mini-agents are preserved. `persist_memory`
defaults to `false`; pipeline jobs never create chat rows. `input` may pass a
structured mapping, and `output.shape` validates the final object.

### `decision`

Asks an `OrchidDecisionModel` to choose one of the declared `labels` and returns
`output.label`, confidence and the raw answers. `branches` maps labels to target
jobs; `fallback` may be a label or a job id. `on_low_confidence` selects
`fallback`, `llm` or `fail`. Branch targets are entered only by the decision's
conditional edge — do not also list the decision in their `depends_on`.

A decision provider referenced only by a pipeline job is not built in v1: bind
it to at least one agent (`decision_model`) or a registered decision point so
the factory constructs it. The example's `advisor` does this for the `route`
provider.

### `map`

Fans out over a JMESPath list (`over`) and runs a template job once per item.
The item is bound to the name in `as`; `item`, `index` and `parent_job` are also
available. Items must be mappings. `concurrency` caps in-flight items;
`on_item_error: skip` records failed items without failing the map. The join
returns `{items, failures, skipped, cancelled}` keyed by `map#index`.

### `subpipeline`

Runs another pipeline as a child run. `input` templates the child input,
`merge_graph` optionally merges its objects into the parent graph, and
`output_from` selects the child value to return (default `output`). Nesting is
bounded by `max_subpipeline_depth`; cycles are rejected at compile time.

### `emit`

Renders a payload and dedupe key, builds an identity-only `SignalEnvelope`
(source `pipeline:<name>`, run correlation id, tenant/user, identity claim) and
hands it to the configured emitter. It never persists the envelope itself and
never places the bearer token in it. If the events layer is disabled the step
fails with `pipeline_events_disabled`; set `on_error: skip` if a pipeline must
also run without events, or install a local emitter (the example's startup hook
does this for direct CLI runs).

### `custom`

Loads an `OrchidPipelineStep` subclass from a dotted path:
`kind: custom`, `class: my_pkg.steps.MyStep`. The class must expose `kind` and
`execute(ctx)`. Prefer the built-in kinds unless you need a new runtime
dependency.

### v1 scheduling note

`depends_on` wires scheduling edges, not an all-predecessor barrier. In v1 a
job with multiple independent predecessors becomes eligible as soon as one
predecessor triggers it; it is not guaranteed to wait for the others. For a
strict sequence, chain jobs with a single predecessor. `join: any` is the
supported convergence point for mutually exclusive decision branches. Use the
example as a reminder: `report` reads the earlier `snapshot` output but
declares only `escalate` as its predecessor.

## Mapping context

Templates and JMESPath expressions evaluate against one documented context:

| Key | Contents |
|-----|----------|
| `input` | The validated run input. |
| `run` | Run identity: `run.id`, `run.pipeline`, `run.mode`, timestamps. |
| `jobs` | Per-job results: `jobs.<id>.output`, `.status`, `.kind`, `.error`. |
| `objects` | Every graph object keyed by id. |
| `types` | Objects grouped by type (useful for `map.over`). |
| `aliases` | Latest object written under each alias. |
| `summary` | The interpretation text, when one already exists. |

`{{ ... }}` templates render a value; a bare JMESPath string (no `{{`) is
evaluated directly for fields documented as JMESPath (`when`, `over`,
`output_from`, `context_from`). Missing paths fail the job rather than silently
rendering an empty string, except where a field documents a default.

## Results and interpretation

A run returns an `OrchidPipelineResult`:

| Field | Meaning |
|-------|---------|
| `run_id`, `pipeline` | Identity of the run and its definition. |
| `status` | `succeeded`, `partial`, `failed` or `cancelled`. A timeout is reported as a cancelled run or a typed error. |
| `output` | The typed output object, or `null` when projection failed. |
| `summary`, `summary_source` | Interpretation text and how it was produced. |
| `jobs` | Per-job `{kind, status, output, duration_ms, llm_calls, prompt_tokens, completion_tokens, total_tokens, attempts, objects, error}`. |
| `graph_stats` | Object/relation/byte counts, per-type counts, LLM usage and errors. |
| `graph` | The serialized graph snapshot when `include_graph` is requested and within the cap. |
| `error` | The fatal error message, if any. |

Interpretation modes:

- `none` — empty summary.
- `caller` — a deterministic sentence built from status, decisions and counts.
- `template` — a `{{ }}` template over `output`, `graph_stats` and `run`.
- `llm` — one bounded summarisation call over the output and graph stats.

The graph snapshot is opt-in (`include_graph` / `--include-graph`) because it can
be large. When it exceeds `max_graph_bytes`, the result reports that the graph
was omitted instead of truncating it silently.

## Limits

Defaults live in `pipelines.limits`; a definition may tighten them.

| Limit | Default | Protects |
|-------|---------|----------|
| `timeout_seconds` | none | Whole-run deadline. |
| `max_jobs` | 100 | Declared jobs. |
| `max_objects` | 10,000 | Graph object count. |
| `max_relations` | 50,000 | Graph relation count. |
| `max_object_bytes` | 1,000,000 | One serialized object. |
| `max_graph_bytes` | 20,000,000 | Serialized graph snapshot. |
| `max_llm_calls` | 50 | LLM calls per run. |
| `max_prompt_chars` | 32,000 | Rendered prompt size. |
| `max_context_chars` | 8,000 | Retrieved/RAG context. |
| `max_map_items` | 100 | Items in one map. |
| `max_parallel` | 8 | Concurrent jobs/items. |
| `max_subpipeline_depth` | 3 | Nested child runs. |

Compilation rejects unknown job kinds, missing dependencies, cycles, invalid
shape references, branch targets that are not real jobs, and a `join: all` job
that depends on mutually exclusive branch targets.

## Facade, CLI and HTTP

### Facade

```python
result = await orchid.run_pipeline(
    "customer_health",
    {"customer_id": "C-100"},
    auth=auth,                 # optional OrchidAuthContext
    seed=[{"id": "account:1"}],
    mode="direct",             # or "bloom"
    correlation_id="req-1",
    chat_id=None,
    timeout=30.0,
    include_graph=True,
    observer=observer,         # optional pipeline event observer
)
```

`stream_pipeline(...)` yields `pipeline.started`, `pipeline.job.*`,
`pipeline.finished` and error events; the final event carries the result.

### CLI

```bash
orchid pipeline list --config orchid.yml --json
orchid pipeline show customer_health --config orchid.yml
orchid pipeline validate customer_health --config orchid.yml --input-json '{"customer_id":"C-1"}'
orchid pipeline run customer_health --config orchid.yml \
  --input-json '{"customer_id":"C-1"}' --output json --include-graph
```

`list`, `show` and `validate` are offline: they load the configuration and
registry without starting MCP, models or tools. `run` uses the normal facade
bootstrap and per-user MCP warm-up. Machine output goes to stdout; verbose
events go to stderr. Exit codes: `0` success, `1` internal error, `2`
validation, `3` failed/partial, `4` timeout, `5` not found.

### HTTP

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/pipelines` | Paginated public list. |
| `GET` | `/pipelines/{name}` | Public definition detail. |
| `POST` | `/pipelines/{name}/validate` | Validate input without running. |
| `POST` | `/pipelines/{name}/runs` | Sync run (`mode: sync`) or Bloom launch (`mode: async`). |
| `POST` | `/pipelines/{name}/runs/stream` | SSE event stream for a sync run. |

Auth uses the same bearer/identity resolver as the rest of the API. Errors are
stable `{error: {code, message, details?}}` bodies. A sync run may return 200
with a failed result; invalid input is 422; a graph snapshot over the cap is
413. Async runs return 202 and a `status_url` under `/jobs/{trigger_id}/runs`;
`seed`, `include_graph` and `timeout_seconds` are sync-only.

## Security notes

- **Auth is execution context.** The pipeline reads `OrchidAuthContext` from the
  run configuration, never from graph state, so it is not checkpointed. Agent,
  tool and RAG steps receive the caller's tenant/user scope.
- **Deterministic steps are trusted code.** `uses` is a dotted import resolved at
  compile time; do not point it at user-controlled strings.
- **Writes are allowlisted.** A job may only write declared shapes and relation
  types. Relation endpoints are templates, and typed relation writes validate
  both ends.
- **Emit carries identity, not credentials.** The envelope contains the signal
  type, JSON payload, source, correlation id, tenant/user identifiers and an
  identity claim; the bearer token is never included. Use `dedupe_key` for
  idempotent consumers.
- **The public detail view is sanitized.** It exposes names, jobs, shapes and
  exposure flags — never prompts, tool names, class paths or object data.
- **Do not log payloads.** Pipeline observability logs statuses, counts and
  bounded errors; prompts and rendered object data stay out of logs.

## See also

- [Data pipeline example](../../examples/data_pipeline/README.md)
- [`README.md`](../README.md) — installation and a quick-start `agents.yaml`
