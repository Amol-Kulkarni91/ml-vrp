"""Train a 2-layer LSTM demand forecast, log to MLflow, compare with LightGBM."""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.features import FEATURE_COLS, TARGET


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep
        last = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last).squeeze(-1)


def prepare_sequences(df: pd.DataFrame, seq_len: int = 7):
    """
    Convert zone-day data into sequences for the LSTM.
    Input:  df with [PULocationID, date, trip_count, *FEATURE_COLS]
    Output: X (batch, seq_len, input_size), y (batch,)
    """
    X_list, y_list = [], []
    for zone, group in df.groupby("PULocationID"):
        g = group.sort_values("date").reset_index(drop=True)
        values = g[FEATURE_COLS + [TARGET]].values.astype(np.float32)
        for i in range(seq_len, len(values)):
            X_list.append(values[i - seq_len:i])   # (seq_len, input_size)
            y_list.append(values[i, -1])            # target = next day's trip_count
    return torch.tensor(X_list), torch.tensor(y_list, dtype=torch.float32)


def train():
    df = pd.read_parquet("data/processed/features.parquet")
    df = df.dropna(subset=FEATURE_COLS + [TARGET])

    # Chronological split: last 14 days = test
    dates = sorted(df["date"].unique())
    test_dates = set(dates[-14:])
    train_mask = ~df["date"].isin(test_dates)

    X_train, y_train = prepare_sequences(df[train_mask], seq_len=7)
    X_test, y_test = prepare_sequences(df[~train_mask], seq_len=7)

    print(f"Train: {len(X_train)} sequences | Test: {len(X_test)} sequences")
    print(f"Input size: {X_train.shape[2]}")

    model = LSTMForecaster(input_size=X_train.shape[2], hidden_size=128, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Train
    model.train()
    batch_size = 256
    num_epochs = 50
    history = []

    for epoch in range(num_epochs):
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg_loss = epoch_loss / len(X_train)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
    y_true = y_test.numpy()

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-8))) * 100

    print(f"\nPyTorch LSTM  →  MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.1f}%")

    # Compare with LightGBM
    try:
        import json
        with open("results/lightgbm_metrics.json") as f:
            lgbm_metrics = json.load(f)
        print(f"LightGBM     →  MAE: {lgbm_metrics['mae']:.2f}  RMSE: {lgbm_metrics['rmse']:.2f}  MAPE: {lgbm_metrics['mape_pct']:.1f}%")
    except FileNotFoundError:
        print("LightGBM metrics not found (run train_lightgbm.py first)")

    # Log to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("ml-vrp")
    with mlflow.start_run(run_name="pytorch-lstm"):
        mlflow.log_params({
            "model": "LSTM",
            "hidden_size": 128,
            "num_layers": 2,
            "seq_len": 7,
            "lr": 1e-3,
            "epochs": num_epochs,
            "batch_size": 256,
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape_pct": mape})
        mlflow.pytorch.log_model(model, "model")

    # Save
    torch.save(model.state_dict(), "results/lstm_forecast.pt")
    print("\nModel saved to results/lstm_forecast.pt")

    # Save predictions for the VRP solver (same format as LightGBM)
    # Map back to zone-day for the latest test dates
    test_df = df[~train_mask].reset_index(drop=True)
    # For the portfolio: just save the aggregate forecast per zone for the latest date
    latest_date = test_df["date"].max()
    latest = test_df[test_df["date"] == latest_date]
    latest_out = latest[["PULocationID", "date"]].copy()
    # Use the model's predictions for the last sequence per zone
    latest_out["forecast"] = preds[-len(latest_out):]
    latest_out.to_json("results/forecast_lstm.json", orient="records")
    print(f"Forecast saved to results/forecast_lstm.json ({len(latest_out)} zones)")


if __name__ == "__main__":
    train()   