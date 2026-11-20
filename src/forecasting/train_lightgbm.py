"""Train LightGBM demand forecast per zone, log to MLflow."""
import json
import os

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.features import FEATURE_COLS, TARGET


def train():
    df = pd.read_parquet("data/processed/features.parquet")

    # Chronological split: last 14 days = test
    dates = sorted(df["date"].unique())
    test_dates = set(dates[-14:])
    train_mask = ~df["date"].isin(test_dates)

    X_train = df[train_mask][FEATURE_COLS]
    y_train = df[train_mask][TARGET]
    X_test = df[~train_mask][FEATURE_COLS]
    y_test = df[~train_mask][TARGET]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    
    preds = model.predict(X_test)
    y = y_test.to_numpy(dtype=float)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    wape = float(np.abs(y - preds).sum() / y.sum() * 100)

    # Naive baselines on the same test rows
    ok = X_test["lag_7"].notna().to_numpy() & X_test["lag_1"].notna().to_numpy()
    naive7_mae = mean_absolute_error(y[ok], X_test["lag_7"].to_numpy()[ok])
    naive1_mae = mean_absolute_error(y[ok], X_test["lag_1"].to_numpy()[ok])
    model_mae_ok = mean_absolute_error(y[ok], preds[ok])

    metrics = {
        "mae": mae, "rmse": rmse, "wape_pct": wape,
        "naive_lag7_mae": naive7_mae, "naive_lag1_mae": naive1_mae,
        "improvement_vs_lag7_pct": (1 - model_mae_ok / naive7_mae) * 100,
        "n_test_rows": int(len(y)),
    }

    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("ml-vrp")
    with mlflow.start_run(run_name="lightgbm-baseline"):
        mlflow.log_params({
            "n_estimators": 300, "learning_rate": 0.05,
            "max_depth": 6, "num_leaves": 31,
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "wape_pct": wape})
        mlflow.lightgbm.log_model(model, "model")

    # Save for serving
    model.booster_.save_model("results/lightgbm_model.txt")
   
    print(f"MAE: {mae:.2f} trips")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.1f}%")
    print("Model saved to results/lightgbm_model.txt")
    
    with open("results/lightgbm_metrics.json", "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "wape_pct": wape}, f)

    # Save predictions for the VRP solver
    out = df[~train_mask][["PULocationID", "date"]].assign(forecast=preds)
    out["PULocationID"] = out["PULocationID"].astype(int)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_json("results/forecast.json", orient="records")




if __name__ == "__main__":
    train() 