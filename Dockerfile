FROM python:3.13-slim
WORKDIR /app

# Install libgomp (required by OR-Tools / LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY . .
CMD uv run uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT}   