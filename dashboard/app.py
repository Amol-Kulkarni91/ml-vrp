"""ML-VRP Dashboard: routes on map, forecast accuracy, drift history."""
import json
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium  
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import folium

st.set_page_config(page_title="ML-VRP Dashboard", layout="wide")
st.title("ML-Driven VRP — Live Dashboard")
st.caption("Daily demand forecast → OR-Tools route optimization → drift monitoring")

RESULTS = Path(__file__).parent.parent / "results"
DATA = Path(__file__).parent.parent / "data" / "processed"

# ─── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_routes():
    path = RESULTS / "routes.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_forecast():
    path = RESULTS / "forecast.json"
    if not path.exists():
        return None
    df = pd.read_json(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_actual():
    path = DATA / "zone_day_demand.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_centroids():
    path = DATA / "zone_centroids.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_drift_history():
    path = RESULTS / "drift.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

routes = load_routes()
forecast = load_forecast()
actual = load_actual()
centroids = load_centroids()
drift = load_drift_history()

# ─── Tab 1: Routes on Map ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Routes", "📈 Forecast Accuracy", "📊 Drift"])

with tab1:
    if routes is None or routes.get("status") != "optimal":
        st.warning("No routes available yet.")
    elif centroids is not None:
        st.metric("Total distance", f"{routes['total_distance_m'] / 1000:.1f} km",
                  delta=f"{routes['num_routes']} routes")

        m = folium.Map(location=[40.75, -73.95], zoom_start=12)

        colors = px.colors.qualitative.Set2
        for i, route in enumerate(routes["routes"]):
            color = colors[i % len(colors)]
            # Get lat/lon for each zone in the route
            points = []
            for zone_idx in route:
                zone_id = routes["zone_ids"][zone_idx]
                row = centroids[centroids["PULocationID"] == zone_id]
                if not row.empty:
                    points.append([row["lat"].values[0], row["lon"].values[0]])
            if len(points) > 1:
                folium.PolyLine(
                    points, color=color, weight=4, opacity=0.8,
                    popup=f"Route {i+1} ({len(route)} stops)"
                ).add_to(m)
                # Mark stops
                for j, pt in enumerate(points):
                    folium.CircleMarker(
                        pt, radius=6, color=color, fill=True,
                        popup=f"Stop {j+1}"
                    ).add_to(m)

        st_folium(m, width=800, height=500)
    else:
        st.info("Zone centroids not found. Run `uv run python -m src.data.zones`.")

with tab2:
    if forecast is not None and actual is not None:
        # Merge forecast vs actual
        merged = forecast.merge(actual, on=["PULocationID", "date"],
                                suffixes=("_pred", "_actual"))
        if len(merged) > 0:
            # MAE over time
            merged["error"] = (merged["forecast"] - merged["trip_count"]).abs()
            daily_mae = merged.groupby("date")["error"].mean().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_mae["date"], y=daily_mae["error"],
                                      mode="lines+markers", name="Daily MAE"))
            fig.update_layout(title="Forecast Error Over Time (MAE)",
                              xaxis_title="Date", yaxis_title="MAE (trips)")
            st.plotly_chart(fig, use_container_width=True)

            # Per-zone comparison (latest date)
            latest = merged["date"].max()
            latest_df = merged[merged["date"] == latest]
            fig2 = px.bar(latest_df, x="PULocationID", y=["forecast", "trip_count"],
                          labels={"value": "Trips", "PULocationID": "Zone"},
                          title=f"Forecast vs Actual — {latest.date()}")
            fig2.update_layout(barmode="group")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No overlapping dates between forecast and actual data.")
    else:
        st.warning("Forecast or actual data not found.")

with tab3:
    if drift is not None:
        st.json(drift)
        if drift.get("status") == "drift_detected":
            st.error(f"⚠️ Drift detected (PSI={drift['psi']}). Retrain triggered.")
        else:
            st.success(f"✅ Model stable (PSI={drift['psi']})")
    else:
        st.info("No drift data yet. Will appear after first daily run.")   