"""Detect forecast drift using PSI. Trigger retrain if threshold exceeded."""
import json
import pandas as pd
import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    < 0.10 = stable
    0.10–0.25 = moderate drift (watch)
    > 0.25 = significant drift (retrain)
    """
    # Normalize both to [0, 1] using expected's range
    e_min, e_max = expected.min(), expected.max()
    span = (e_max - e_min) + 1e-8
    e_norm = (expected - e_min) / span
    a_norm = (actual - e_min) / span

    # Bin edges
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    e_counts, _ = np.histogram(e_norm, breakpoints)
    a_counts, _ = np.histogram(a_norm, breakpoints)

    # Convert to proportions, clip to avoid log(0)
    e_prop = np.clip(e_counts / len(expected), 1e-4, None)
    a_prop = np.clip(a_counts / len(actual), 1e-4, None)

    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def check_drift(
    forecast_file: str = "results/forecast.json",
    actual_file: str = "data/processed/zone_day_demand.parquet",
    lookback_days: int = 7,
    threshold: float = 0.25,
) -> dict:
    """
    Compare the last N days of model forecast against actual demand.
    Returns PSI and a retrain recommendation.
    """
    forecast = pd.read_json(forecast_file)
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["PULocationID"] = forecast["PULocationID"].astype(int)
    actual = pd.read_parquet(actual_file)
    actual["date"] = pd.to_datetime(actual["date"])
    actual["PULocationID"] = actual["PULocationID"].astype(int)

    # Use the latest date in forecast as "today"
    latest = pd.Timestamp(forecast["date"].max())
    window_start = latest - pd.Timedelta(days=int(lookback_days))   

    # Filter both to the overlap window
    recent_forecast = forecast[forecast["date"] >= window_start]
    recent_actual = actual[actual["date"] >= window_start]

    # Merge on zone × date
    merged = recent_forecast.merge(
        recent_actual,
        on=["PULocationID", "date"],
        suffixes=("_pred", "_actual"),
    )

    if len(merged) < 10:
        return {"psi": None, "status": "insufficient_data", "retrain": False,
                "n_samples": len(merged)}

    psi = compute_psi(
        merged["forecast"].values.astype(float),
        merged["trip_count"].values.astype(float),
    )

    status = "drift_detected" if psi > threshold else "stable"
    result = {
        "psi": round(psi, 4),
        "threshold": threshold,
        "status": status,
        "retrain": psi > threshold,
        "n_samples": len(merged),
        "lookback_days": lookback_days,
        "date": str(latest.date()),
    }

    print(f"Drift check ({latest.date()}): PSI={psi:.4f} "
          f"(threshold={threshold}) → {status}")
    return result


if __name__ == "__main__":
    result = check_drift()
    print(json.dumps(result, indent=2))   