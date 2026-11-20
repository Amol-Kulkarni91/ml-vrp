"""Feature engineering for zone-day demand forecasting."""

import pandas as pd


def build_features(df: pd.DataFrame, zone_ids=None) -> pd.DataFrame:
    """
    Input: df with [PULocationID, date, trip_count]
    Output: complete zone x day grid (missing days = 0 trips) with
            lag, rolling (past-only) and calendar features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.groupby(["PULocationID", "date"], as_index=False)["trip_count"].sum()

    if zone_ids is None:
        zone_ids = sorted(df["PULocationID"].unique())
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([zone_ids, dates], names=["PULocationID", "date"])
    df = (
        df.set_index(["PULocationID", "date"])["trip_count"]
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    features = []
    for zone, g in df.groupby("PULocationID"):
        g = g.sort_values("date").reset_index(drop=True)
        past = g["trip_count"].shift(1)  # exclude today
        g["lag_1"] = past
        g["lag_7"] = g["trip_count"].shift(7)
        g["rolling_mean_7"] = past.rolling(7).mean()
        g["rolling_std_7"] = past.rolling(7).std()
        g["day_of_week"] = g["date"].dt.dayofweek
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        features.append(g)

    out = pd.concat(features, ignore_index=True)
    return out.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"])


FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "is_weekend",
]
TARGET = "trip_count"

if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    demand = pd.read_parquet("data/processed/zone_day_demand.parquet")
    feat = build_features(demand, cfg["data"]["zone_ids"])
    feat.to_parquet("data/processed/features.parquet", index=False)
    print(f"Feature matrix: {feat.shape}")
    print(feat[FEATURE_COLS].describe())
