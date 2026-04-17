# ADR-0005: Deployment Strategy — Fly.io Prototypes and GKE Production Path

**Date:** 2026-04-14
**Status:** Accepted
**Deciders:** Ben Lin

---

## Context

The project needed a deployment strategy that satisfied two distinct, concurrent requirements:

1. **Short-lived prototypes** — rapid iteration deployments from a developer's local machine with minimal infrastructure overhead.
2. **Production path** — a CI/CD pipeline that will eventually deploy to Google Kubernetes Engine (GKE) via Azure DevOps (ADO) pipelines, JFrog Artifactory as the image registry, and a separate GitOps system for cluster reconciliation.

The constraints:
- No external image registry during the prototype phase (images stay local or in Fly.io's internal registry).
- The pipeline structure must not require a rewrite when the production infrastructure becomes available — the migration should be additive.
- Helm configuration must co-locate with the application source so it migrates to ADO intact.

---

## Decision

### Prototype deployment (now)

Prototypes are deployed from a local machine using two shell scripts:

- **`scripts/fly-setup.sh`** — idempotent bootstrap of a named Fly.io app and its Postgres cluster. Safe to re-run; skips already-created resources.
- **`scripts/fly-deploy.sh`** — runs unit tests, then builds the Docker image locally (`flyctl deploy --local-only`) and pushes it to Fly's internal registry for that app. No external registry is involved.

`fly.toml` configures the Fly.io machine: `shared-cpu-1x` with 512 MB RAM, rolling deploy strategy, health checks against `GET /health`, and scale-to-zero when idle.

Multiple named prototype instances can coexist using the `[app-name]` argument to both scripts, which overrides the `app` field in `fly.toml`.

### CI (GitHub Actions, now)

GitHub Actions runs unit tests on every push for coverage statistics. It does **not** build Docker images or deploy — those remain local operations during the prototype phase.

The workflow (`ci.yml`) uses `uv sync --frozen` and `uv run` throughout, consistent with the project's `uv`-only constraint (see CLAUDE.md).

### Production pipeline (future — Azure DevOps)

When the project migrates to ADO with JFrog and GKE, the pipeline shape is:

```
push to deploy branch
  → Stage: Test    (uv sync, uv run pytest)
  → Stage: Build   (docker build, docker push → JFrog, capture digest)
  → Stage: Publish (update helm/lithium/values.yaml image.digest, commit)
```

A separate GitOps system (e.g. ArgoCD) watches the repository and reconciles GKE state from the Helm chart when `values.yaml` changes.

### Helm chart

`helm/lithium/` is scaffolded now so it migrates to ADO with the rest of the source. Key design choices:

- The `image.digest` field in `values.yaml` is the handoff point between the ADO pipeline and the deployment system. The pipeline writes the digest; the deployment system reads it.
- When `image.digest` is set it takes precedence over `image.tag` in the Deployment template, ensuring exact-image deployments regardless of mutable tag behaviour.
- A pod annotation (`checksum/image-digest`) forces a rollout whenever the digest changes, even if the tag is unchanged.
- Secrets (`ANTHROPIC_API_KEY`, `DATABASE_URL`) are referenced from a cluster Secret (`lithium-secrets`) — never baked into `values.yaml`.

---

## Rationale

### Why `flyctl deploy --local-only` rather than Fly's remote builder

`--local-only` uses the developer's local Docker daemon. This keeps the prototype flow entirely self-contained: the only external dependency is a Fly.io account. It also means the Dockerfile is exercised locally before any remote operation, catching build failures earlier.

### Why no image registry during the prototype phase

Attaching a registry now would require managing credentials, repository hygiene, and image lifecycle for images that will be discarded. The cost exceeds the benefit for short-lived work.

### Why scaffold the Helm chart now

The Helm chart will need to exist in the ADO repository. Since the ADO migration copies this repository, creating `helm/lithium/` now ensures it arrives with the codebase rather than being created in isolation later — reducing the risk of the chart diverging from the application it deploys.

### Why the pipeline writes `values.yaml` rather than calling `kubectl` or `helm upgrade` directly

Treating `values.yaml` as the deployment artefact decouples the pipeline from the cluster. The pipeline only needs write access to the repository; cluster credentials stay with the GitOps system. This also gives a full audit trail: every image digest that was ever deployed is in git history.

---

## Consequences

**Positive**

- Prototype cycle is: `./scripts/fly-setup.sh && ./scripts/fly-deploy.sh` — two commands, no registry, no pipeline.
- The ADO pipeline migration is additive: add a Build stage that pushes to JFrog and writes `values.yaml`. The Test stage and Helm chart are already in place.
- Multiple prototype instances can coexist (`fly-setup.sh lithium-exp-1`, `fly-setup.sh lithium-exp-2`) without conflicting.
- Fly.io scales to zero when idle, so dormant prototypes do not incur compute cost.

**Negative / trade-offs**

- `flyctl deploy --local-only` requires Docker to be running on the developer's machine. The Fly.io remote builder is not used, so build-environment differences between developers are possible.
- The Helm chart is scaffolded without a live GKE cluster to validate against. It should be smoke-tested (`helm template`, `helm lint`) before the first real GKE deploy.
- The prototype workflow bypasses CI — a developer can deploy without tests passing. `fly-deploy.sh` enforces unit tests, but this is a convention, not a gate.
