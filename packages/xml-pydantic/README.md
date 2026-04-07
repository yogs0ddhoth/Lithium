# xml-pydantic

A library for bidirectional conversion between XML, JSON Schema, and Pydantic v2 models. It lets you author XML files that serve simultaneously as human-readable documentation and as machine-readable schemas, then generate strongly-typed Pydantic models from them at runtime.

## How it works

```
XML file (data-* attributes as schema hints)
    ↓  parse_xml_file / parse_xml_string
JSON Schema dict
    ↓  define_model
Pydantic BaseModel class  ←→  model_to_xml_string  →  XML string
```

`data-*` attributes on XML elements carry JSON Schema keywords in kebab-case HTML style, which are automatically converted to the camelCase form required by JSON Schema:

```xml
<Person data-type="object">
  <name data-type="string" data-required="true">Full legal name.</name>
  <age  data-type="integer" data-minimum="0" data-maximum="150" />
</Person>
```

## Installation

This package is not published to PyPI. Install it as a local path dependency using uv.

**In another local project's `pyproject.toml`:**

```toml
[project]
dependencies = [
    "xml-pydantic>=0.1.0",
]

[tool.uv.sources]
xml-pydantic = { path = "../xml-pydantic", editable = true }
```

Then sync:

```bash
uv sync
```

**Or install directly into an environment:**

```bash
uv add --editable /path/to/packages/xml-pydantic
```

## Usage

```python
from xml_pydantic import define_model, parse_xml_file, model_to_xml_string

# Parse an XML schema file into a JSON Schema dict
schema = parse_xml_file("my_schema.xml")

# Generate a Pydantic model class from the schema
MyModel = define_model("MyModel", schema)

# Instantiate and validate
instance = MyModel(name="Alice", age=30)

# Serialize back to XML
xml_str = model_to_xml_string(instance)
```

### Public API

| Symbol | Description |
| --- | --- |
| `parse_xml_file(path)` | Parse an XML file → JSON Schema dict |
| `parse_xml_string(xml)` | Parse an XML string → JSON Schema dict |
| `element_to_schema(element)` | Convert an `ET.Element` → JSON Schema dict |
| `define_model(name, schema)` | Generate a Pydantic `BaseModel` class from a JSON Schema dict |
| `model_to_xml(model)` | Convert a Pydantic model instance → `ET.Element` |
| `model_to_xml_string(model)` | Convert a Pydantic model instance → XML string |
| `dict_to_xml(data, root_tag)` | Convert a plain dict → `ET.Element` |

## Development

### Setup

```bash
# From the packages/xml-pydantic/ directory
uv sync
```

This creates a `.venv` inside `packages/xml-pydantic/` with all runtime and dev dependencies installed.

### Testing

```bash
uv run pytest
```

Run a single test file:

```bash
uv run pytest tests/test_parse_xml.py
```

Run with verbose output:

```bash
uv run pytest -v
```

Coverage is reported automatically via `pytest-cov`. The configuration in `pyproject.toml` targets `xml_pydantic` as the coverage source and prints a term-missing report after each run.

### Linting and formatting

This package inherits the ruff configuration from the root `pyproject.toml` when run from the repo root. To lint within the package directory:

```bash
# From repo root
uv run ruff check --fix packages/xml-pydantic/
uv run ruff format packages/xml-pydantic/
```

### Adding dependencies

```bash
uv add <package>            # runtime dependency
uv add --dev <package>      # dev-only dependency (added to [dependency-groups].dev)
```

## Building

To produce a distributable wheel or sdist:

```bash
# From the packages/xml-pydantic/ directory
uv build
```

Output artifacts are placed in `packages/xml-pydantic/dist/`:

```
dist/
  xml_pydantic-0.1.0-py3-none-any.whl
  xml_pydantic-0.1.0.tar.gz
```

To install the built wheel into another environment without uv path sources:

```bash
pip install dist/xml_pydantic-0.1.0-py3-none-any.whl
# or
uv add dist/xml_pydantic-0.1.0-py3-none-any.whl
```

### Versioning

Update the `version` field in `pyproject.toml` before building. No release automation is configured — increment the version manually following [semver](https://semver.org/) conventions.
