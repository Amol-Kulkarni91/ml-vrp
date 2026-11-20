"""Train a 2-layer LSTM demand forecast, log to MLflow."""
import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from src.data.features import FEATURE_COLS, TARGET

SEED, SEQ_LEN = 42, 7
COUNT_COLS = ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7", TARGET]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def prepare_sequences(df, test_dates, zone_scale, seq_len=SEQ_LEN):
    """Build sequences over each zone's full history and split by TARGET date,
    so test targets still get their 7 days of past context. Count-like columns
    are divided by the zone's training mean."""
    cols = FEATURE_COLS + [TARGET]
    Xtr, ytr, Xte, yte, meta = [], [], [], [], []
    for zone, group in df.groupby("PULocationID"):
        g = group.sort_values("date").reset_index(drop=True)
        scale = zone_scale[zone]
        vals = g[cols].copy()
        vals[COUNT_COLS] = vals[COUNT_COLS] / scale
        vals = vals.values.astype(np.float32)
        for i in range(seq_len, len(g)):
            x, y, d = vals[i - seq_len:i], vals[i, -1], g.loc[i, "date"]
            if d in test_dates:
                Xte.append(x); yte.append(y); meta.append((int(zone), d, scale))
            else:
                Xtr.append(x); ytr.append(y)
    t = lambda a: torch.tensor(np.stack(a))
    return (t(Xtr), torch.tensor(ytr, dtype=torch.float32),
            t(Xte), torch.tensor(yte, dtype=torch.float32), meta)


def train():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = pd.read_parquet("data/processed/features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET])

    dates = sorted(pd.to_datetime(df["date"].unique()))
    test_dates = set(dates[-14:])
    train_df = df[~df["date"].isin(test_dates)]
    zone_scale = (train_df.groupby("PULocationID")[TARGET].mean() + 1.0).to_dict()

    X_train, y_train, X_test, y_test, meta = prepare_sequences(df, test_dates, zone_scale)
    print(f"Train: {len(X_train)} sequences | Test: {len(X_test)} sequences")

    model = LSTMForecaster(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    num_epochs, batch_size = 50, 256

    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            loss = criterion(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  loss={epoch_loss/len(X_train):.4f}")

    model.eval()
    scales = np.array([m[2] for m in meta], dtype=np.float32)
    with torch.no_grad():
        preds = np.clip(model(X_test).numpy() * scales, 0, None)
    y_true = y_test.numpy() * scales

    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    wape = float(np.abs(y_true - preds).sum() / y_true.sum() * 100)
    metrics = {"mae": mae, "rmse": rmse, "wape_pct": wape, "n_test_rows": len(y_true)}
    print(f"PyTorch LSTM -> {metrics}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("ml-vrp")
    with mlflow.start_run(run_name="pytorch-lstm"):
        mlflow.log_params({"model": "LSTM", "hidden_size": 128, "num_layers": 2,
                           "seq_len": SEQ_LEN, "lr": 1e-3, "epochs": num_epochs,
                           "scaling": "per-zone train mean"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n_test_rows"})

    torch.save(model.state_dict(), "results/lstm_forecast.pt")
    with open("results/lstm_metrics.json", "w") as f:
        json.dump(metrics, f)

    out = pd.DataFrame({
        "PULocationID": [m[0] for m in meta],
        "date": [m[1].strftime("%Y-%m-%d") for m in meta],
        "forecast": preds,
    })
    out.to_json("results/forecast_lstm.json", orient="records")


if __name__ == "__main__":
    train()