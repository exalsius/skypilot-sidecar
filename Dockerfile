# -----------------------------
# Stage 1: Base build with uv
# -----------------------------
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS builder

WORKDIR /app

# Install build-deps, sync production deps, then clean up
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# -----------------------------
# Stage 2: Run tests
# -----------------------------
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS tester

WORKDIR /app

COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH"

# -----------------------------
# Stage 3: Final minimal service
# -----------------------------
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS server

WORKDIR /app

COPY --from=tester --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
