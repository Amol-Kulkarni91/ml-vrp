"""Download NYC TLC Yellow Taxi Parquet files and aggregate to zone-day demand."""
import os
import pandas as pd
import requests
from pathlib import Path
from src.config import load_config

config = load_config()


def download_month(month: str, dest: str = "data/raw/") -> Path:
    """Download a single month's parquet file from NYC TLC."""
    os.makedirs(dest, exist_ok=True)
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month[:4]}-{month[5:]}.parquet"
    path = Path(dest) / f"yellow_tripdata_{month}.parquet"
    if path.exists():
        return path
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
    return path

def load_and_aggregate(months: list[str], zone_ids: list[int]) -> pd.DataFrame:
    """
    Load monthly parquets, filter to target zones,
    aggregate to zone × day trip counts.
    """
    dfs = []
    for month in months:
        path = download_month(month)
        df = pd.read_parquet(
            path,
            columns=["tpep_pickup_datetime", "PULocationID"],
            filters=[("PULocationID", "in", zone_ids)],
        )
        dfs.append(df)

    trips = pd.concat(dfs, ignore_index=True)
    trips["date"] = pd.to_datetime(trips["tpep_pickup_datetime"]).dt.date
    trips["hour"] = pd.to_datetime(trips["tpep_pickup_datetime"]).dt.hour

    # Aggregate: zone × day → trip count
    demand = (
        trips.groupby(["PULocationID", "date"])
        .size()
        .reset_index(name="trip_count")
        .sort_values(["PULocationID", "date"])
        .reset_index(drop=True)
    )
    return demand

if __name__ == "__main__":
    cfg = load_config()
    demand = load_and_aggregate(cfg["data"]["months"], cfg["data"]["zone_ids"])
    demand.to_parquet("data/processed/zone_day_demand.parquet", index=False)
    print(f"Saved {len(demand)} rows to data/processed/zone_day_demand.parquet")
    print(demand.head(20))