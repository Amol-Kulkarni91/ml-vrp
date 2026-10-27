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

- **API:** [Endpoint](https://ml-vrp.onrender.com/docs)
- **Dashboard:** [Simple Dashboard](https://ml-vrp-1.onrender.com/)

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
```
## Results

| Model | MAE (trips) | RMSE | MAPE |
|-------|:-----------:|:----:|:----:|
| LightGBM | 14.65 | 29.27 | 23.48 |
| PyTorch LSTM (2-layer, 128 hidden) | 101.9 | 42.77 | 190.33 |


| Metric | Value |
|--------|-------|
| Zones | 18 (NYC, zones 23–42) |
| Vehicles | 20 |
| Total route distance | 273.4 km |
| Solver | OR-Tools (PATH_CHEAPEST_ARC, 30s time limit) |
| Distance metric | Haversine (great-circle) from zone centroids |



| Layer | Tool |
|-------|------|
| Forecasting | LightGBM, PyTorch (LSTM) |
| Optimization | OR-Tools (CVRP with time windows) |
| MLOps | MLflow (experiment tracking), PSI drift monitoring, auto-retrain |
| Serving | FastAPI, Uvicorn, Render |
| Orchestration | GitHub Actions (daily cron + keep-alive) |
| Dashboard | Streamlit, Plotly, Folium, Hugging Face Spaces |
| Data | NYC TLC Yellow Taxi (6 months, 18 zones) |
| Config | Pydantic, YAML |
| Packaging | uv, Docker |

## Repo Structure
```
├── src/
│   ├── data/              # Ingestion, feature engineering, zone centroids
│   ├── forecasting/       # LightGBM + PyTorch training scripts
│   ├── optimization/      # OR-Tools VRP solver (haversine)
│   ├── serving/           # FastAPI endpoints
│   ├── monitoring/        # PSI drift detection
│   └── pipeline/          # Airflow DAG (local dev)
├── dashboard/             # Streamlit app (deployed to HF Space)
├── results/               # Committed outputs (forecast, routes, drift, model)
├── configs/               # YAML configuration
├── tests/                 # Unit tests + Locust load test
├── .github/workflows/     # Daily pipeline, keep-alive, CI
├── pyproject.toml         # uv-managed dependencies
├── uv.lock                # Locked dependency tree
├── Dockerfile             # Production image (Render)
└── docker-compose.yml     # Local dev (MLflow) 
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM over LSTM for production | GBM won on MAE for tabular time-series. Simpler, faster to train, easier to debug. |
| Haversine over road-network distances | Documented simplification. Production would use OSRM/GraphHopper for travel times. |
| Pre-computed results in the API | The API reads JSON committed by the daily pipeline. Avoids running the solver per-request. |
| `/predict` for live inference | Single-point prediction for demos and what-if analysis. Not used in the daily pipeline. |
| PSI over KS test for drift | PSI is more interpretable for multi-modal distributions; well-established thresholds (0.10 / 0.25). |
| uv over pip/requirements.txt | Reproducible installs via `uv.lock`. Faster resolution. Single tool for venv + deps + running. |
| Render over FastAPI Cloud | Needs `libgomp1` (system lib) for LightGBM. FastAPI Cloud doesn't support custom system deps. |

## Daily Pipeline
06:00 UTC ──▶ Ingest ──▶ Features ──▶ Train ──▶ Solve ──▶ Drift ──▶ Commit
                                                                    │
                                                       if PSI > 0.25
                                                                    ▼
                                                              Retrain   

## Limitations
- Haversine distances underestimate real travel time (no road network, no traffic).
- Single visit per zone (no repeated visits for high-demand zones).
- 18 zones is a small instance; production would need partitioning or a different solver.
- No GPU training (LSTM is CPU-only, small enough for this scale).




---

## Where to find each `[FILL IN]`

| Placeholder | Where to get it |
|---|---|
| LightGBM MAE / RMSE / MAPE | Output of `uv run python -m src.forecasting.train_lightgbm` (the printed metrics) |
| LSTM MAE / RMSE / MAPE | Output of `uv run python -m src.forecasting.train_pytorch` |
| Vehicles | `configs/config.yaml` → `optimization.num_vehicles` |
| Total route distance | Output of `uv run python -m src.optimization.vrp_solver` (the "Total distance" line) |
| p95 latencies | Output of `uv run locust ...` (the summary table at the end) |
| Render URL | Your actual Render dashboard URL |
| HF Space URL | Your actual Hugging Face Space URL |

**Quick way to grab all the numbers at once:**

```bash
# Forecasting
uv run python -m src.forecasting.train_lightgbm 2>&1 | grep -E "MAE|RMSE|MAPE"
uv run python -m src.forecasting.train_pytorch 2>&1 | grep -E "MAE|RMSE|MAPE"

# Routes
uv run python -m src.optimization.vrp_solver 2>&1 | grep "Total distance"

# Latency (run once, copy the table)
uv run locust -f tests/load_test.py --headless -u 50 -r 10 -t 60s 2>&1 | tail -10   