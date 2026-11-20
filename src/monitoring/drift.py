"""Detect forecast drift using PSI. Trigger retrain if threshold exceeded."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 5) -> float:
    """PSI with quantile bins from the expected distribution.
    <0.10 stable, 0.10-0.25 watch, >0.25 retrain."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, edges)
    a_counts, _ = np.histogram(actual, edges)
    e_prop = np.clip(e_counts / len(expected), 1e-4, None)
    a_prop = np.clip(a_counts / len(actual), 1e-4, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def check_drift(
    forecast_file="results/forecast.json",
    actual_file="data/processed/zone_day_demand.parquet",
    lookback_days=7,
    threshold=0.25,
    min_samples=10,
) -> dict:
    forecast = pd.read_json(forecast_file)
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["PULocationID"] = forecast["PULocationID"].astype(int)
    actual = pd.read_parquet(actual_file)
    actual["date"] = pd.to_datetime(actual["date"])
    actual["PULocationID"] = actual["PULocationID"].astype(int)
    actual = actual[["PULocationID", "date", "trip_count"]]

    run_date = str(pd.Timestamp.today().date())

    # Only dates where BOTH a forecast and an actual exist
    merged = forecast.merge(actual, on=["PULocationID", "date"], how="inner")
    if merged.empty:
        return {
            "run_date": run_date,
            "psi": None,
            "status": "no_overlap",
            "retrain": False,
            "n_samples": 0,
            "forecast_max_date": str(forecast["date"].max().date()),
            "actual_max_date": str(actual["date"].max().date()),
        }

    latest = merged["date"].max()
    merged = merged[merged["date"] > latest - pd.Timedelta(days=lookback_days)]
    if len(merged) < min_samples:
        return {
            "run_date": run_date,
            "psi": None,
            "status": "insufficient_data",
            "retrain": False,
            "n_samples": len(merged),
        }

    # Seasonal-naive baseline: same zone, same weekday last week
    naive = actual.copy()
    naive["date"] = naive["date"] + pd.Timedelta(days=7)
    naive = naive.rename(columns={"trip_count": "naive_7d"})
    merged = merged.merge(naive, on=["PULocationID", "date"], how="left")

    psi = compute_psi(merged["forecast"].values, merged["trip_count"].values)
    if not np.isfinite(psi):
        return {
            "run_date": run_date,
            "psi": None,
            "status": "insufficient_data",
            "retrain": False,
            "n_samples": len(merged),
        }

    err = (merged["forecast"] - merged["trip_count"]).abs()
    has_naive = merged["naive_7d"].notna()
    mae = float(err[has_naive].mean()) if has_naive.any() else float(err.mean())
    naive_mae = (
        float(
            (merged.loc[has_naive, "naive_7d"] - merged.loc[has_naive, "trip_count"])
            .abs()
            .mean()
        )
        if has_naive.any()
        else None
    )

    status = "drift_detected" if psi > threshold else "stable"
    result = {
        "run_date": run_date,
        "date": str(latest.date()),
        "psi": round(psi, 4),
        "threshold": threshold,
        "status": status,
        "retrain": psi > threshold,
        "n_samples": len(merged),
        "lookback_days": lookback_days,
        "mae": round(mae, 2),
        "naive_mae": round(naive_mae, 2) if naive_mae is not None else None,
    }
    print(f"Drift check ({latest.date()}): PSI={psi:.4f} -> {status}", file=sys.stderr)
    return result


if __name__ == "__main__":
    import json
    from pathlib import Path

    result = check_drift()
    Path("results").mkdir(exist_ok=True)

    # Latest result, for the dashboard
    Path("results/drift.json").write_text(json.dumps(result, indent=2))

    # History, one entry per run date
    hist_path = Path("results/drift_history.json")
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    history = [h for h in history if h.get("run_date") != result.get("run_date")] + [
        result
    ]
    hist_path.write_text(json.dumps(history, indent=2))

    print(json.dumps(result, indent=2))
