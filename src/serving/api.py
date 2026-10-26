"""FastAPI service: serves latest forecast and routes."""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML-VRP API", version="0.1.0")

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

# Replace with:
_booster = None

def _get_booster():
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file=str(RESULTS / "lightgbm_model.txt"))
    return _booster

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
    
class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    rolling_mean_7: float
    rolling_std_7: float
    day_of_week: int
    is_weekend: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast", response_model=list[ForecastResponse])
def get_forecast():
    """Return the latest forecast (pre-computed by GitHub Actions)."""
    path = RESULTS / "forecast.json"
    if not path.exists():
        return []
    df = pd.read_json(path)
    df["date"] = pd.to_datetime(df["date"])
    latest = df["date"].max()
    today = df[df["date"] == latest]
    return [
        {"date": r["date"].strftime("%Y-%m-%d"), "zone_id": int(r["PULocationID"]), "forecast": float(r["forecast"])}
        for _, r in today.iterrows()
    ]   

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
    from config import load_config
    from optimization.vrp_solver import solve_vrp
    cfg = load_config()
    forecast = pd.read_json(RESULTS / "forecast.json")
    latest = forecast["date"].max()
    today = forecast[forecast["date"] == latest]
    result = solve_vrp(today, cfg)
    with open(RESULTS / "routes.json", "w") as f:
        json.dump(result, f, indent=2)
    return result   

@app.post("/predict", response_model=dict)
def predict(req: PredictRequest):
    """Live inference: pass features, get forecast."""
    X = pd.DataFrame([[req.lag_1, req.lag_7, req.rolling_mean_7,
                       req.rolling_std_7, req.day_of_week, req.is_weekend]],
                     columns=["lag_1", "lag_7", "rolling_mean_7",
                              "rolling_std_7", "day_of_week", "is_weekend"])
    pred = _get_booster().predict(X)[0]
    return {"forecast": round(float(pred), 2)}

@app.get("/debug")
def debug():
    import os
    return {
        "cwd": os.getcwd(),
        "results_exists": RESULTS.exists(),
        "results_files": [f.name for f in RESULTS.iterdir()] if RESULTS.exists() else [],
        "abs_path": str(RESULTS.resolve()),
    }   
    
    
