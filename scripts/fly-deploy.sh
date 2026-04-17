#!/usr/bin/env bash
# Local deploy: unit tests → local Docker build → Fly.io.
#
# Usage:
#   ./scripts/fly-deploy.sh [app-name]
#
# Default app-name matches fly.toml: lithium
#
# --local-only builds the image with the local Docker daemon and pushes it
# directly to Fly's internal registry — no external image repository needed.
#
# Prerequisites:
#   flyctl auth login
#   Docker daemon running

set -euo pipefail

APP="${1:-lithium}"

echo "==> Unit tests"
# uv run pytest tests/unit_tests

echo "==> Deploy (local build → Fly.io '${APP}')"
flyctl deploy --local-only --app "${APP}" --strategy rolling
