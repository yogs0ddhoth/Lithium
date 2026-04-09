# ADR-0001: XML as the Dual-Purpose Prompt and Schema Format

**Date:** 2026-03-25
**Status:** Accepted
**Deciders:** Ben Lin

---

## Context

The Orient agent produces structured data at two boundaries:

1. **Tool output** — each tool invocation must hand off a well-formed data structure to the next step (agent or tool).
2. **Final output** — the agent produces a `<problem_statement />` document consumed by external systems.

The initial prototype (commit `a4b73bb`) used ad-hoc Python classes in `src/orient/models.py` to define output schemas. These were coupled to the prompt text: the LLM instruction and the expected data shape were maintained separately, so they could drift.

The team also needed a way to iterate on both the agent's behaviour (the prompt) and its output contract (the schema) simultaneously, without a code–prompt synchronisation step.

---

## Decision

Prompt files in `prompts/` are XML documents that serve **two simultaneous roles**:

1. **LLM system prompt** — the XML content is parsed into a string and injected into the model's system message.
2. **Pydantic model schema** — `data-type`, `data-required`, and `data-description` attributes on XML elements are read by `xml-pydantic` to generate a Pydantic v2 model class at runtime.

A standalone local package (`packages/xml-pydantic/`) handles the bidirectional conversion: XML schema → JSON Schema → `datamodel-code-generator` → Pydantic model class. The same package serialises model instances back to XML for tool-to-tool handoffs.

The dynamic models are constructed once at import time in `app/orient/prompts.py` and injected into the tools via LangGraph's `ToolRuntime`.

---

## Rationale

### Alternatives considered

**JSON Schema files alongside prompts**
Each prompt would have a paired `.json` schema. Rejected because it doubles the number of files to keep in sync and loses the co-location benefit — a prompt change still requires a separate schema edit.

**Pydantic models defined in Python code**
The original approach. Rejected because the schema becomes invisible to the LLM prompt: if the prompt says "return a `<qa_results>` block" and the code expects a different shape, there is no single source of truth to catch the mismatch at authoring time.

**JSON-based tool communication**
Standard LangChain tool pattern. Rejected because the agent is already structured around XML prompts; using JSON for tool I/O would require format translation at each boundary and lose the human-readable, diff-friendly properties of XML for prompt engineering work.

### Why XML over other markup

- The LLM instruction and the schema annotation live on the same node — changing a field name updates both simultaneously.
- XML attributes (`data-type`, `data-required`) are unambiguous extension points that do not pollute the semantic content.
- XML diffs are legible in code review in a way that JSON Schema diffs often are not.

---

## Consequences

**Positive**
- A single file change updates both the LLM instruction and the enforced output contract.
- Pydantic validation catches malformed LLM output before it reaches downstream tools.
- The `xml-pydantic` library is independently testable and reusable (394 tests added in commit `8b28ee1`).

**Negative / trade-offs**
- Runtime code generation (`datamodel-code-generator`) adds import-time overhead and a dependency that is unusual in production Python services.
- The `prompts.py` dynamic model types are not statically analysable — mypy reports errors on `QAResults` and `ProblemStatement` because they are variables, not type aliases. This is a known limitation noted in the codebase.
- Developers unfamiliar with `data-*` attributes must learn the convention before editing prompt files.

---

## Addendum — 2026-04-08: `XmlDto` mixin for serialisation

### Context

As DTO classes multiplied from 2 (Orient) to 9+ (Converge/Diverge), every class carried an identical `model_dump_xml` method body differing only in the `root_tag` string. This is mechanical repetition with no per-class logic — a maintenance hazard and a sign that `root_tag` belongs in the class definition, not repeated in each method body.

### Decision

A `XmlDto` mixin class lives in `app/utils.py`. It declares a `_root_tag: str` class variable and provides the single canonical `model_dump_xml` implementation. Every DTO class uses multiple inheritance to combine the dynamic Pydantic model with the mixin:

```python
class MyDto(xml_pydantic.define_model("MyDto", SCHEMA), XmlDto):
    """DTO for `<my_element />`."""
    _root_tag = "my_element"
```

The mixin is placed second in the MRO so the dynamic Pydantic model's `__init__` and field machinery take precedence. `_root_tag` starts with an underscore, which Pydantic v2 treats as a private name and excludes from field introspection.

### Rationale

- The `root_tag` value is a property of the DTO class's XML contract, not of each call site — declaring it as a class variable is the correct level of abstraction.
- Placing the mixin in `app/utils.py` (already imported by both `prompts.py` modules) avoids a new shared module without creating a dependency cycle.
- The pattern scales: adding a new DTO is a 3-line change (`SCHEMA = ...`, `class Foo(..., XmlDto): _root_tag = ...`).

### Consequences

- All DTO classes across `app/orient/prompts.py` and `app/converge_diverge/prompts.py` use this pattern. A class without `XmlDto` is a bug.
- `# type: ignore[arg-type]` is present on the `model_to_xml_string` call inside `XmlDto` because `self` is typed as the mixin, not as a Pydantic model — this is the only location where the static-analysis gap from the original decision surfaces in application code.
