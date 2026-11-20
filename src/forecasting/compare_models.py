"""Compare LightGBM, LSTM and naive baselines on identical test rows."""

import json

import pandas as pd

from src.data.features import TARGET


def load(path: str, name: str) -> pd.DataFrame:
    d = pd.read_json(path)
    d["date"] = pd.to_datetime(d["date"])
    d["PULocationID"] = d["PULocationID"].astype(int)
    return d.rename(columns={"forecast": name})


def compare() -> dict:
    f = pd.read_parquet("data/processed/features.parquet")
    f["date"] = pd.to_datetime(f["date"])
    m = f.merge(
        load("results/forecast.json", "lgbm"), on=["PULocationID", "date"]
    ).merge(load("results/forecast_lstm.json", "lstm"), on=["PULocationID", "date"])

    results = {}
    for name, col in [
        ("LightGBM", "lgbm"),
        ("LSTM", "lstm"),
        ("Naive lag-7", "lag_7"),
        ("Naive lag-1", "lag_1"),
    ]:
        e = m[col] - m[TARGET]
        results[name] = {
            "mae": round(float(e.abs().mean()), 2),
            "rmse": round(float((e**2).mean() ** 0.5), 2),
            "wape_pct": round(float(e.abs().sum() / m[TARGET].sum() * 100), 1),
            "n": len(m),
        }
        print(f"{name:12s} " + "  ".join(f"{k}={v}" for k, v in results[name].items()))
    return results


if __name__ == "__main__":
    out = compare()
    with open("results/model_comparison.json", "w") as fh:
        json.dump(out, fh, indent=2)
