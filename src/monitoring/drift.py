# src/monitoring/drift.py
"""Compare recent actual demand vs. model forecast using PSI."""
import pandas as pd
import numpy as np

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. < 0.1 = stable, > 0.25 = significant drift."""
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    # Normalize to [0,1] for binning
    e_min, e_max = expected.min(), max(expected.max(), actual.max())
    e_norm = (expected - e_min) / (e_max - e_min + 1e-8)
    a_norm = (actual - e_min) / (e_max - e_min + 1e-8)
    e_bins = np.histogram(e_norm, breakpoints)[0] / len(expected)
    a_bins = np.histogram(a_norm, breakpoints)[0] / len(actual)
    # Avoid log(0)
    e_bins = np.clip(e_bins, 1e-4, None)
    a_bins = np.clip(a_bins, 1e-4, None)
    return float(np.sum((a_bins - e_bins) * np.log(a_bins / e_bins)))

def check_drift(forecast_file: str = "results/forecast.json",
                actual_file: str = "data/processed/zone_day_demand.parquet",
                threshold: float = 0.25) -> dict:
    """Compare last 7 days of forecast vs. actual. Return PSI + recommendation."""
    forecast = pd.read_json(forecast_file)
    actual = pd.read_parquet(actual_file)
    actual["date"] = pd.to_datetime(actual["date"])

    # Last 7 days overlap
    latest = forecast["date"].max()
    recent = forecast[forecast["date"] >= pd.Timestamp(latest) - pd.Timedelta(days=7)]
    actual_recent = actual[actual["date"] >= pd.Timestamp(latest) - pd.Timedelta(days=7)]

    # Merge on zone × date
    merged = recent.merge(actual_recent, on=["PULocationID", "date"], suffixes=("_pred", "_actual"))
    if len(merged) == 0:
        return {"psi": None, "status": "no_overlap"}

    psi = compute_psi(merged["forecast"].values, merged["trip_count"].values)
    status = "drift_detected" if psi > threshold else "stable"
    return {"psi": round(psi, 4), "threshold": threshold, "status": status,
            "retrain": psi > threshold}   