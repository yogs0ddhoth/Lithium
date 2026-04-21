# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Always use `uv run` to execute Python code and tools. Never call `python`, `pytest`, `ruff`, or other tools directly — they may not resolve to the project's virtual environment.

### Setup

```bash
uv sync  # Install all dependencies
```

### Running

```bash
langgraph dev        # Run the LangGraph app with hot reload (dev only)
uvicorn app.main:app --host 0.0.0.0 --port 8000  # Production server
```

### Testing

- `uv run pytest` — Run unit tests (with coverage)
- `uv run pytest tests/unit_tests/test_foo.py` — Run a specific test file

### Linting and Formatting

- `uv run ruff check .` — Lint
- `uv run ruff check --fix .` — Lint and auto-fix
- `uv run ruff format .` — Format code
- `uv run ruff format --check .` — Check formatting

### Type Checking

- `uv run ty check` or `uv run mypy .`

### Dependencies

- `uv add <package>` — Add a dependency
- `uv add --dev <package>` — Add a dev dependency
- `uv remove <package>` — Remove a dependency
- `uv sync` — Sync environment
- `uv lock` — Update lockfile

## Package Management Constraints

This project uses uv exclusively. These constraints are strict:

- All Python dependencies **must** be installed, synchronized, and locked using uv
- Never use `pip`, `pip-tools`, `poetry`, or `conda` for dependency management
- Do not create or activate virtual environments manually — uv manages `.venv/` automatically
- Do not install packages globally or with `pip install`
- Do not create `requirements.txt` — use `pyproject.toml` and `uv.lock`
- Do not add dependencies to `pyproject.toml` by hand — use `uv add`
- Do not create `setup.py`, `setup.cfg`, or `requirements.txt`
- Always use `pyproject.toml` for metadata (PEP 621)

## Architecture

**Entry point**: `app/main.py` dual-exports:

- `orient` — the compiled LangGraph graph (referenced by `langgraph.json` for `langgraph dev`)
- `app` — the FastAPI ASGI application (used by `uvicorn` in production)

### Core Components

1. **Orient Agent** (`app/orient/`) — ReAct loop that validates, refines, and synthesises problem statements via XML-structured tool calls
2. **Converge/Diverge Agent** (`app/converge_diverge/`) — Sequential four-tool workflow (features → concept scores → feature scores → RTC-EBC) with human-in-the-loop interrupts after the first two tools only (see ADR-0004)
3. **FastAPI Server** (`app/server/`) — Custom production server hosting all agents under a unified `AGENT_REGISTRY`; replaces the LangGraph platform (see ADR-0003)
4. **`xml-pydantic` package** (`packages/xml-pydantic/`) — Local library that converts XML files into JSON Schema dicts and dynamically generates Pydantic v2 models; also serialises model instances back to XML
5. **Prompts** (`prompts/`) — XML files that serve as both LLM system prompts and Pydantic model schemas (see ADR-0001)

### Orient Agent (`app/orient/`)

Implements a ReAct (Reasoning + Action) cycle (see ADR-0002):

```text
START → call_llm → route_model_output → tools → call_llm → … → END
```

- **`graph.py`** — Defines the LangGraph `StateGraph`. The `call_llm` node invokes the configured LLM; `route_model_output` routes to `tools` (ToolNode) if there are tool calls, otherwise ends. Exports `builder` (uncompiled) and `graph` (compiled with `MemorySaver`).
- **`state.py`** — `InputState` holds messages; `State` extends it with `is_last_step` (managed `IsLastStep`) and `qa_results`.
- **`context.py`** — `Context` / `Configuration` (a `RunnableConfig`-compatible dataclass) controls `model`, prompts, and `max_search_results`. Values can be overridden via uppercase environment variables.
- **`tools.py`** — Two LLM-powered async tools: `review_user_problem` (validates input → `QAResults` XML) and `synthesize_problem_statement` (produces the final `ProblemStatement` XML). Both use `ToolRuntime[Context, State]` to access config and state without global state.
- **`prompts.py`** — Loads XML prompt templates from `prompts/`, calls `xml_pydantic.schema.from_file()` + `define_model()` to build `QAResults` and `ProblemStatement` Pydantic classes at import time. Note: these dynamic types are not statically analysable by mypy — this is a known trade-off (ADR-0001).

### Converge/Diverge Agent (`app/converge_diverge/`)

Runs four LLM-powered tools in sequence with human checkpoints after the first two tools (see ADR-0004):

```text
START → call_llm ──(tool calls)──→ tools ──(map/score_concepts)──→ human_review → call_llm
                 └──(no tool calls)──→ END        └──(other tools)──────────────→ call_llm
```

Tools run in order: `map_feature_to_themes` → `score_concepts` → `score_features` → `generate_rct_ebc`.

- **`graph.py`** — Defines the `StateGraph`. A `human_review` node (using `interrupt()`) sits between `tools` and `call_llm` for `map_feature_to_themes` and `score_concepts` only; `score_features` and `generate_rct_ebc` route directly back to `call_llm` via `route_after_tools`. The LLM is guided by `Context.system_prompt` (loaded from `prompts/converge_diverge.xml`) to call tools in order and pass XML verbatim.
- **`state.py`** — `InputState` holds messages; `State` extends it with `is_last_step`, `features_and_themes`, `concept_scores`, `feature_scores`, and `rtc_ebc`.
- **`context.py`** — Per-tool system prompts and shared LLM config, following the same env-var override pattern as Orient.
- **`tools.py`** — Four async tools, each invoking a sub-LLM with a structured-output schema and writing both to state and to the message thread as XML via `model_dump_xml()`.
- **`prompts.py`** — Loads XML prompt/schema files from `prompts/` and builds dynamic Pydantic models at import time using `xml_pydantic`.

### FastAPI Server (`app/server/`)

Custom production server replacing the LangGraph platform (see ADR-0003):

- **`lifespan.py`** — Defines `AGENT_REGISTRY` (maps agent names → `AgentSpec`). Each `AgentSpec` carries a `context_factory: Callable[[str, dict], Any]` built by `_make_context_factory(ContextCls)`, which reflects over the dataclass fields and ignores unknown keys. Compiles all agents at startup and populates both `app.state.graphs` and `app.state.specs`. Add new agents here.
- **`routes.py`** — `APIRouter` at `/agents/{agent_name}/...`. Provides thread creation, synchronous run/resume (`POST .../runs`), streaming run/resume (`POST .../runs/stream`), and thread state retrieval (`GET .../threads/{thread_id}`). New runs and interrupt resumes share the same two endpoints, distinguished by the `RunInput` body. Context is built via the per-agent factory in `app.state.specs`.
- **`models.py`** — Pydantic request/response schemas (`RunInput`, `RunResponse`, `SSEEvent`, `ThreadResponse`, `ThreadStateResponse`). `RunInput` is a discriminated body: set `message` for a new run or `resume` for an interrupt resume — a `model_validator` enforces mutual exclusivity.
- **`dependencies.py`** — FastAPI dependencies for API key auth and graph resolution from `app.state.graphs`.

**Checkpointer rules:**

- `MemorySaver` (default): in-process, state lost on restart. Must not be used with more than one uvicorn worker.
- `AsyncPostgresSaver`: persistent, requires `DATABASE_URL`. Required for multi-worker deployments.

### `xml-pydantic` Package (`packages/xml-pydantic/`)

Standalone local library implementing the XML ↔ Pydantic bidirectional contract (ADR-0001):

- **`schema.py`** — Parses XML elements using `data-*` attribute conventions into JSON Schema dicts. Entry points: `from_element()`, `from_string()`, `from_file()`.
  - `data-type` → JSON Schema `type`
  - `data-required="true"` → adds field to parent `required[]`
  - `data-description` → JSON Schema `description`
  - `data-min-length`, `data-exclusive-maximum`, etc. → mapped numeric/boolean keywords (kebab-case → camelCase)
- **`serializers.py`** — Serialises Pydantic model instances back to XML. Handles pluralisation/singularisation for list fields (e.g. `items` → `<item>` elements).
- **`__init__.py`** — Exports `define_model(name, schema)` which takes a JSON Schema dict and returns a dynamically constructed Pydantic v2 `BaseModel` subclass via `datamodel-code-generator`.

**Known limitation:** Runtime code generation adds import-time overhead. Dynamic model types are not statically analysable.

**Anti-pattern:** Do not define output schemas as plain Python classes alongside prompts. The schema and prompt must co-locate in a single XML file so they cannot drift (ADR-0001).

### DTO Serialisation Pattern (`app/utils.XmlDto`)

Every DTO class in a `prompts.py` module uses multiple inheritance to combine a dynamic Pydantic model with the `XmlDto` mixin from `app/utils.py`:

```python
class MyDto(xml_pydantic.define_model("MyDto", SCHEMA), XmlDto):
    """DTO for `<my_element />`."""
    _root_tag = "my_element"
```

`XmlDto` inherits from `ABC` and declares `model_dump` as `@abstractmethod`. The Pydantic co-base satisfies that requirement at class creation time — Python's ABC machinery rejects any concrete subclass that omits a Pydantic co-base, regardless of MRO ordering. `_root_tag` is the only class-level customisation required. See ADR-0001 addenda (2026-04-08, 2026-04-20).

**Anti-pattern:** Do not define `model_dump_xml` inline on a DTO class — it belongs in `XmlDto`. A DTO without `XmlDto` in its bases is a bug.

### Prompts (`prompts/`)

XML files serving dual purpose: LLM system prompts **and** Pydantic model schemas. The `data-*` attributes on XML elements drive the output contract without a separate schema file.

**Anti-pattern:** Do not add a paired `.json` schema file alongside a prompt — this breaks co-location and creates two files to keep in sync (ADR-0001).

### Architecture Decision Records (`docs/adr/`)

Authoritative records for non-obvious design choices. Consult before changing core patterns:

- **[ADR-0001](docs/adr/0001-xml-driven-structured-output.md)** — XML as the dual-purpose prompt and schema format *(addendum 2026-04-08: `XmlDto` mixin; addendum 2026-04-20: ABC + Protocol type safety)*
- **[ADR-0002](docs/adr/0002-langgraph-react-over-custom-loop.md)** — LangGraph ReAct pattern over a custom agent loop
- **[ADR-0003](docs/adr/0003-fastapi-server-over-langgraph-platform.md)** — Custom FastAPI server over LangGraph platform deployment *(addendum 2026-04-09: LangGraph v2 output API + unified run/resume endpoint)*
- **[ADR-0004](docs/adr/0004-human-in-the-loop-interrupts.md)** — Human-in-the-loop interrupts for sequential tool workflows *(addendum 2026-04-09: unified run/resume endpoint)*
- **[ADR-0005](docs/adr/0005-deployment-strategy.md)** — Deployment strategy: Fly.io prototypes, GitHub Actions CI, and GKE production path via Azure DevOps + JFrog

**ADR addendum vs new ADR:** Add a dated `## Addendum` section to an existing ADR when the change refines or extends the same architectural decision (e.g. an implementation detail of the same pattern). Create a new numbered ADR only when a distinct, independent decision is being made. Addenda follow the same `### Context / Decision / Rationale / Consequences` structure as the parent record.

## Deployment

### Prototype workflow (`fly.toml`, `scripts/`)

Short-lived prototypes are deployed from a local machine. Two scripts handle the full lifecycle:

- **`scripts/fly-setup.sh [app-name] [region]`** — idempotent bootstrap: creates the Fly.io app, Postgres cluster (`<app-name>-db`), attaches Postgres, and sets secrets. Safe to re-run.
- **`scripts/fly-deploy.sh [app-name]`** — runs unit tests, then `flyctl deploy --local-only` to build the image with the local Docker daemon and push it to Fly's internal registry. No external image registry is used.

`fly.toml` configures the Fly.io machine. The `app` field is the default name; pass an explicit name to both scripts to run isolated prototype instances in parallel.

Secrets (`ANTHROPIC_API_KEY`, `DATABASE_URL`) are set via `fly-setup.sh` and `flyctl postgres attach` — never in `fly.toml` or committed files.

### CI (GitHub Actions)

`.github/workflows/ci.yml` runs lint, format check, and unit tests on every push and pull request. It does **not** build Docker images or deploy — those are local operations during the prototype phase.

`ci.yml` uses `uv sync --frozen` and `uv run` exclusively, consistent with the uv-only constraint.

### Helm chart (`helm/lithium/`)

`helm/lithium/` is the source of truth for GKE cluster configuration. It co-locates with the application source so it migrates to Azure DevOps with the rest of the repository.

- `values.yaml` — default values; `image.digest` and `image.repository` are written by the CI/CD pipeline on each deploy. Do not set these by hand.
- `templates/deployment.yaml` — uses `image.digest` when set (exact-image deploy); falls back to `image.tag`.
- A pod annotation (`checksum/image-digest`) forces a rollout whenever the digest changes.
- Secrets (`ANTHROPIC_API_KEY`, `DATABASE_URL`) are read from the cluster Secret named by `secretName` — never stored in `values.yaml`.

The future ADO pipeline writes `image.digest` to `values.yaml` and commits it; a separate GitOps system (e.g. ArgoCD) reconciles GKE state from that commit. See ADR-0005.

## Environment Variables

Copy `.env.example` to `.env`. Required keys:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Default LLM provider |
| `OPENAI_API_KEY` | Optional alternative provider |
| `LANGSMITH_API_KEY` | Tracing/observability (optional) |
| `CHECKPOINTER` | `memory` (default) or `postgres` |
| `DATABASE_URL` | Required when `CHECKPOINTER=postgres` |

Default model: `anthropic/claude-sonnet-4-5-20250929` (overridable via `MODEL` env var or LangGraph config).

## Testing Patterns

### Directory structure

The test tree mirrors `app/` exactly. When adding tests, place them in the subdirectory that matches the source module:

```
tests/
  unit_tests/
    conftest.py                  ← session-scoped anyio_backend fixture
    test_utils.py                ← app/utils.py
    orient/
      test_context.py            ← app/orient/context.py
      test_graph.py              ← app/orient/graph.py  (route_model_output)
      test_tools.py              ← app/orient/tools.py
    converge_diverge/
      test_context.py            ← app/converge_diverge/context.py
      test_graph.py              ← app/converge_diverge/graph.py  (route_model_output, route_after_tools)
    server/
      test_models.py             ← app/server/models.py  (RunInput validation)
      test_lifespan.py           ← app/server/lifespan.py (_make_context_factory, checkpointer selection)
      test_routes.py             ← app/server/routes.py  (route helpers + HTTP integration)
```

- Test files named `test_*.py`; test functions named `test_*`
- No `__init__.py` in any test directory; `--import-mode=importlib` (set in `pyproject.toml`) allows same-named files across subdirectories
- Import source modules directly from their defining submodule — never through a re-export in `app/server/__init__.py`
- `packages/xml-pydantic/` has its own `tests/` directory and is tested independently

### Do not test dependency behaviour

Unit tests must cover *our* logic, not the correctness of external packages or local libraries we consume. A line that simply delegates to a dependency is covered by that dependency's own test suite; adding a test here only tightens coupling to its interface without verifying anything we own.

**Missing coverage on the following is acceptable and expected:**

- `call_llm` nodes — the body is `model.ainvoke(...)`. Reaching those lines means mocking LangChain's model and asserting it returns a value: that tests LangChain's contract, not ours.
- `human_review` nodes — the body is `interrupt({...})`. `interrupt()` is a LangGraph checkpoint primitive; there is no application logic of our own to verify.
- Tool bodies in `converge_diverge/tools.py` — each tool is `model.ainvoke()` followed by Pydantic `model_validate()` and `Command` construction. The LLM call and structured-output parsing are dependency behaviour. The resulting `Command` shape mirrors what `orient/tools.py` already tests at 100%.
- Graph builder / compile calls — module-level `StateGraph(...)` and `.compile()` test that LangGraph initialises a graph, not our routing or business rules.
- `app/main.py` — creates the FastAPI `app` and the compiled LangGraph `graph`. Tests here verify that LangGraph + FastAPI start up correctly, which is dependency behaviour.
- Postgres checkpointer path (`lifespan.py` lines 143–149) — calls `AsyncPostgresSaver.from_conn_string()`; tests LangGraph's checkpointer, not our agent compilation.
- `define_model` error branch (`xml_pydantic/__init__.py` line 40–41) — the `case not_model: raise ValueError(...)` branch can only be reached by mocking `datamodel-code-generator` to return a non-`BaseModel` class.
- Defensive `if not f.init: continue` guards in `Context.__post_init__` — no field in any `Context` class uses `init=False`; this branch is dead code that guards against a hypothetical future change.

**Anti-patterns to avoid:**

- Do not mock `model.ainvoke` to test a `call_llm` node — if the only assertion is "we called ainvoke with these args and returned its output," the test adds no value.
- Do not mock `interrupt()` to test a `human_review` node — the function has no branching logic of its own.
- Do not assert the internal structure of a dependency's serialised output (e.g. checking that `AIMessage.model_dump()` contains `"type": "ai"`) — test the behaviour of *our* code around it instead.
- Do not add a test whose only purpose is to execute a line that delegates entirely to a dependency; prefer leaving that line uncovered over writing a test that verifies nothing about our code.

### Coverage requirements at each public boundary

When you add or modify a component, the suite must cover its public boundary before the work is complete. Try to only call the necessary tests to confirm that the modified components are covered:

- **New agent in `AGENT_REGISTRY`**: add a lifespan test verifying it is compiled and its `spec.context_factory` is callable; add an `interrupt-schemas` test for its `interrupt_dtos` map.
- **New or changed route**: add HTTP tests for happy-path, auth failure (401), unknown-agent 404, and any new body-validation rules (422).
- **New `RunInput` field or `model_validator` change**: add unit tests covering all valid permutations and every invalid combination that should raise.
- **New `XmlDto` subclass**: verify `model_dump_xml()` produces the expected root tag (not the class name); this fails silently when `_root_tag` is defined without `ClassVar` in the mixin.
- **New prompt path** (`load_xml_prompt` / `xml_pydantic.schema.from_file`): the path must be relative to the project root (the CWD when `uv run pytest` runs); add or update a test that imports the module to confirm no `FileNotFoundError` at import time.
- **New helper in `routes.py`** (`_last_tool_name`, `_resolve_resume`, etc.): add pure unit tests covering empty/None inputs, the happy path, and edge cases — do not rely solely on HTTP-layer tests to cover these.

## Docstrings

Docstrings must accurately represent the code they describe. When a discrepancy is found between a docstring and the actual implementation — during any task, not only docstring-focused ones — fix the docstring immediately as part of that change.

This applies to:

- Function and method signatures (parameters, return types, raised exceptions)
- Module-level docstrings that describe exports or entry points
- Class docstrings that describe behaviour or responsibilities

Do not leave a known-incorrect docstring in place because it is outside the stated scope of a task.

## Code Style

- Follow ruff's defaults: 88-char line length, double quotes, spaces
- Import sorting handled by ruff (`isort` rules via `select = ["I"]`)
- Do not add `# type: ignore` comments without an error code
- Ruff targets Python 3.13, uses Google-style docstrings, enforces pydocstyle `D401` (imperative mood for first lines)
- All agent/tool functions are `async`
- Structured data is exchanged as XML strings (not JSON) using `model_dump_xml()` on dynamically generated Pydantic models

## Pre-commit Hooks

- `uvx prek install` — Install prek hooks (preferred)
- `uvx pre-commit install` — Install pre-commit hooks
- Do not install pre-commit or prek with pip. Use `uvx`.

## Markdown Linting

Ignore markdown linting warnings (e.g. MD032 blanks-around-lists). Do not add blank lines solely to satisfy a linter.

## What NOT to do

- Do not create or activate virtual environments manually
- Do not install packages globally or with `pip install`
- Do not create `requirements.txt` — use `pyproject.toml` and `uv.lock`
- Do not add dependencies to `pyproject.toml` by hand — use `uv add`
- Do not run `python setup.py` commands
- Avoid linting the entire project — focus on singular files when debugging errors
- Do not define output schemas as Python classes separate from their prompts (ADR-0001)
- Do not define `model_dump_xml` inline on a DTO class — use the `XmlDto` mixin from `app.utils` (ADR-0001 addendum)
- Do not use JSON for tool-to-tool communication — use XML strings via `model_dump_xml()` (ADR-0001)
- Do not hard-code a single agent into server routes — register new agents in `AGENT_REGISTRY` (ADR-0003)
- Do not hard-code a single `Context` class in routes — use the per-agent `context_factory` from `app.state.specs` (ADR-0004)
- Do not add a new `AgentSpec` without an explicit `context_factory` — the default silently builds the Orient context for any agent (ADR-0004)
- Do not use `interrupt_after=["tools"]` at compile time for human feedback workflows — use a dedicated `human_review` node with `interrupt()` so feedback can be injected as a `HumanMessage` (ADR-0004)
- Do not add separate `/runs/resume` or `/runs/resume/stream` endpoints — new runs and interrupt resumes share `POST .../runs` and `POST .../runs/stream` via `RunInput.message` / `RunInput.resume` (ADR-0003 addendum, ADR-0004 addendum)
- Do not call `ainvoke`/`astream` without `version="v2"` — v2 activates typed state coercion via `_output_mapper`; access `ainvoke` results via `.messages` (attribute), not `["messages"]` (dict key)
- Do not change the `Any` annotation on `ainvoke` results to a concrete type without also adding an explicit `output_schema` to the `StateGraph` builder — the type checker cannot see the v2 dataclass coercion
- Do not run `MemorySaver` with more than one uvicorn worker — use `CHECKPOINTER=postgres` instead
- Do not set `image.digest` or `image.repository` in `helm/lithium/values.yaml` by hand — these are written by the CI/CD pipeline (ADR-0005)
- Do not commit secrets to `fly.toml` or `values.yaml` — use `flyctl secrets set` for Fly.io and a cluster Secret for GKE (ADR-0005)
- Do not use `flyctl deploy` without `--local-only` for prototype deploys — the remote builder bypasses local verification and is not part of the established workflow (ADR-0005)
- Do not add a GitHub Actions deploy workflow — prototype deploys are local; the future deploy pipeline lives in Azure DevOps (ADR-0005)

## Keeping This File Up to Date

When you introduce a new component, establish a new architectural pattern, or discover an anti-pattern, you **must** propose an update to this file. Do not apply the update automatically.

The workflow is:

1. Draft the proposed addition or change (new component description, pattern, anti-pattern, or ADR entry).
2. Present the draft to the human and explain what changed and why.
3. Apply the edit **only after the human explicitly approves**.

This applies to:

- New agents or modules added under `app/`
- New packages added under `packages/`
- New or updated ADRs in `docs/adr/` (including addenda to existing ADRs)
- Architectural patterns or anti-patterns discovered during implementation
- Changes to tooling, deployment, or environment variable conventions
