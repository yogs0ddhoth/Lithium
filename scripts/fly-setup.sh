#!/usr/bin/env bash
# Idempotent Fly.io environment setup.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-ant-... ./scripts/fly-setup.sh [app-name] [region]
#
# Defaults: app-name=lithium  region=iad
#
# Safe to re-run — existing apps, Postgres clusters, and secrets are not
# recreated or overwritten (secrets are upserted by flyctl).
#
# Prerequisites:
#   flyctl auth login

set -euo pipefail

APP="${1:-lithium}"
REGION="${2:-iad}"
PG="${APP}-db"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Error: ANTHROPIC_API_KEY must be set in the environment." >&2
  exit 1
fi

echo "==> App: ${APP}  Region: ${REGION}"

# ── App ───────────────────────────────────────────────────────────────────────
if flyctl apps list --json | grep -q "\"Name\":\"${APP}\""; then
  echo "    App '${APP}' already exists, skipping creation."
else
  echo "    Creating app '${APP}'..."
  flyctl apps create "${APP}" --machines
fi

# ── Postgres ──────────────────────────────────────────────────────────────────
if flyctl apps list --json | grep -q "\"Name\":\"${PG}\""; then
  echo "    Postgres cluster '${PG}' already exists, skipping creation."
else
  echo "    Creating Postgres cluster '${PG}'..."
  flyctl postgres create \
    --name "${PG}" \
    --region "${REGION}" \
    --initial-cluster-size 1 \
    --vm-size shared-cpu-1x \
    --volume-size 1
fi

# Attach sets DATABASE_URL as a Fly secret automatically.
# Tolerate the case where it is already attached.
flyctl postgres attach "${PG}" --app "${APP}" 2>/dev/null \
  || echo "    Postgres already attached to '${APP}'."

# ── Secrets ───────────────────────────────────────────────────────────────────
echo "==> Setting secrets..."
flyctl secrets set ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" --app "${APP}"

echo ""
echo "Setup complete for '${APP}'."
echo ""
echo "Next steps:"
echo "  Deploy:   ./scripts/fly-deploy.sh ${APP}"
echo "  Teardown: flyctl apps destroy ${APP} --yes && flyctl apps destroy ${PG} --yes"
