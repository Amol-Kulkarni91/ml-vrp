# ML-Driven VRP: Demand Forecasting + Route Optimization

End-to-end system that forecasts per-zone delivery demand using gradient-boosted
and neural models, then solves a capacitated VRP with time windows via OR-Tools,
served as a FastAPI endpoint with full MLOps (tracking, monitoring, auto-retraining).

## Architecture
┌─────────────────────────────────────────────────────────────────────┐
│ GitHub Actions (daily, 06:00 UTC) │
│ │
│ 1. Ingest NYC TLC ──▶ 2. Forecast ──▶ 3. Solve VRP ──▶ 4. Drift │
│ (6 months, 18 zones) LightGBM OR-Tools CVRP-TW PSI │
│ + PyTorch LSTM (haversine) check │
│ │
│ If PSI > 0.25 ──▶ Retrain ──▶ Re-solve ──▶ Commit results │
└─────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ Render (always-on) │
│ FastAPI: /forecast /routes /predict /health │
└─────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ Hugging Face Space (Streamlit dashboard) │
│ Routes on map · Forecast accuracy · Drift history │
└─────────────────────────────────────────────────────────────────────┘



## Live Demo

- **API:** [https://YOUR-PROJECT.onrender.com/docs](https://YOUR-PROJECT.onrender.com/docs)
- **Dashboard:** [https://YOUR-SPACE.hf.space](https://YOUR-SPACE.hf.space)

## Quick Start

```bash
uv sync
uv run python -m src.data.ingest       # Download 6 months of NYC TLC data
uv run python -m src.data.zones        # Extract zone centroids
uv run python -m src.data.features     # Build lag/rolling features
uv run python -m src.forecasting.train_lightgbm  # Train + log to MLflow
uv run python -m src.forecasting.train_pytorch   # Train LSTM + compare
uv run python -m src.optimization.vrp_solver     # Solve CVRP-TW
uv run python -m src.monitoring.drift            # Check PSI
uv run uvicorn src.serving.api:app --reload --port 8000   