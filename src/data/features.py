"""Feature engineering for zone-day demand forecasting."""
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with columns [PULocationID, date, trip_count]
    Output: df with lag, rolling, and calendar features per zone.
    """
    df = df.sort_values(["PULocationID", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    features = []
    for zone, group in df.groupby("PULocationID"):
        g = group.reset_index(drop=True)
        g["lag_1"] = g["trip_count"].shift(1)
        g["lag_7"] = g["trip_count"].shift(7)
        g["rolling_mean_7"] = g["trip_count"].rolling(7).mean()
        g["rolling_std_7"] = g["trip_count"].rolling(7).std()
        g["day_of_week"] = g["date"].dt.dayofweek
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        features.append(g)

    out = pd.concat(features, ignore_index=True)
    out = out.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"])
    return out

FEATURE_COLS = ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7",
                "day_of_week", "is_weekend"]
TARGET = "trip_count"

if __name__ == "__main__":
    demand = pd.read_parquet("data/processed/zone_day_demand.parquet")
    feat = build_features(demand)
    feat.to_parquet("data/processed/features.parquet", index=False)
    print(f"Feature matrix: {feat.shape}")
    print(feat[FEATURE_COLS].describe())   