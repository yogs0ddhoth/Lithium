# Project Name

Brief description of what the project does.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python -m project_name
```

## Development

```bash
# Development workflow
uv run pytest                    # Run tests
uv run ruff check --fix .       # Lint and auto-fix
uv run ruff format .            # Format code

# Dependency management
uv add <package_name>             # Add runtime dependency
uv add --dev <package_name>       # Add dev dependency
uv remove <package_name>          # Remove dependency
uv sync                         # Sync environment with lockfile
uv lock                         # Update lockfile

# Project execution
uv run python main.py         # Run script
uv run python -m <module>         # Run module
uv run --with package cmd       # Run with temporary dependency

# One-off tool execution
uvx pre-commit install          # Install pre-commit hooks
uvx black .                     # Run black without adding to project
```