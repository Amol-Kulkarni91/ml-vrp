"""FastAPI service: serves latest forecast and routes."""
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb

app = FastAPI(title="ML-VRP API", version="0.1.0")
_booster = lgb.Booster(model_file="results/lightgbm_model.txt")


RESULTS = Path("results")

class ForecastResponse(BaseModel):
    date: str
    zone_id: int
    forecast: float

class RoutesResponse(BaseModel):
    status: str
    num_routes: int
    total_distance_m: int
    routes: list[list[int]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast", response_model=list[ForecastResponse])
def get_forecast():
    """Return the latest forecast (pre-computed by Airflow/GitHub Actions)."""
    path = RESULTS / "forecast.json"
    if not path.exists():
        return []
    df = pd.read_json(path)
    latest = df["date"].max()
    today = df[df["date"] == latest]
    return today[["date", "PULocationID", "forecast"]].to_dict("records")

@app.get("/routes", response_model=RoutesResponse)
def get_routes():
    """Return the latest solved routes."""
    path = RESULTS / "routes.json"
    if not path.exists():
        return RoutesResponse(status="no_solution", num_routes=0, total_distance_m=0, routes=[])
    with open(path) as f:
        return json.load(f)

@app.post("/solve")
def trigger_solve():
    """Trigger a re-solve (for local dev; in production this is the Airflow DAG)."""
    from optimization.vrp_solver import solve_vrp
    from config import load_config
    cfg = load_config()
    forecast = pd.read_json(RESULTS / "forecast.json")
    latest = forecast["date"].max()
    today = forecast[forecast["date"] == latest]
    result = solve_vrp(today, cfg)
    with open(RESULTS / "routes.json", "w") as f:
        json.dump(result, f, indent=2)
    return result   


@app.get("/debug")
def debug():
    import os
    return {
        "cwd": os.getcwd(),
        "results_exists": RESULTS.exists(),
        "results_files": [f.name for f in RESULTS.iterdir()] if RESULTS.exists() else [],
        "abs_path": str(RESULTS.resolve()),
    }   