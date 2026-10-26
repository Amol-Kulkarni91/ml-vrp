import zipfile
import io
import os
import tempfile
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.config import load_config

SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

def download_zone_centroids(zone_ids: list[int]) -> pd.DataFrame:
    """Download the official TLC shapefile, compute centroids for target zones."""
    print("Downloading taxi_zones.zip...")
    r = requests.get(SHAPEFILE_URL, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extractall(tmpdir)
            # Find the .shp file (may be in a subfolder)
            shp_path = os.path.join(tmpdir, [n for n in z.namelist() if n.endswith(".shp")][0])
            gdf = gpd.read_file(shp_path)

    gdf = gdf[gdf["LocationID"].isin(zone_ids)].copy()

    # Reproject to WGS84 (lat/lon in degrees)
    gdf = gdf.to_crs(epsg=4326)

    centroids = gdf.geometry.centroid
    result = pd.DataFrame({
        "PULocationID": gdf["LocationID"].values,
        "lat": centroids.y.values,
        "lon": centroids.x.values,
    }).sort_values("PULocationID").reset_index(drop=True)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    result.to_csv("data/processed/zone_centroids.csv", index=False)
    print(f"Saved {len(result)} zone centroids:")
    print(result.to_string())
    return result


if __name__ == "__main__":
    cfg = load_config()
    download_zone_centroids(cfg["data"]["zone_ids"])   