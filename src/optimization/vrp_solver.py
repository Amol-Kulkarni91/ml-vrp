"""Solve CVRP-TW using OR-Tools with forecasted demand as node weights."""
import json
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from src.config import load_config

def build_distance_matrix(zone_ids: list[int]) -> list[list[int]]:
    """
    Haversine distance in meters between zones.
    In production, replace with OSRM/GraphHopper travel times.
    For the portfolio, haversine is fine and documented.
    """
    # Zone centroids (lat, lon) — pull from taxi_zones.geojson
    # For now, use a small hardcoded subset or load from a CSV
    # TODO: load from data/processed/zone_centroids.csv
    raise NotImplementedError("Load zone centroids from taxi_zones data")

def solve_vrp(forecast: pd.DataFrame, cfg: dict) -> dict:
    """
    forecast: DataFrame with [PULocationID, forecast] for today
    cfg: config dict with optimization params
    """
    demand = forecast.set_index("PULocationID")["forecast"].astype(int).tolist()
    zone_ids = forecast["PULocationID"].tolist()
    num_locations = len(zone_ids)
    num_vehicles = cfg["optimization"]["num_vehicles"]
    capacity = cfg["optimization"]["vehicle_capacity"]
    tw = cfg["optimization"]["time_window"]

    # Distance matrix (placeholder — replace with real distances)
    # Using a simple synthetic matrix for the skeleton
    np.random.seed(42)
    dist_matrix = np.random.randint(1000, 5000, size=(num_locations, num_locations))
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) // 2  # symmetrize
    dist_matrix = dist_matrix.tolist()

    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(
        num_locations, num_vehicles, 0  # depot = zone 0
    )
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def dist_cb(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # Capacity dimension
    def demand_cb(from_index):
        return demand[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    demand = forecast.set_index("PULocationID")["forecast"].astype(int).tolist()
    demand[0] = 0  # depot doesn't consume capacity
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [capacity] * num_vehicles, True, "Capacity"
    )

    # Time window dimension
    def time_cb(from_index, to_index):
        # Approximate: distance / 30 km/h → minutes
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] // 500

    time_idx = routing.RegisterTransitCallback(time_cb)
    tw_rel = [tw[0], tw[1] - tw[0]]  # [0, 1200]
    
    routing.AddDimension(time_idx, 60, tw[1] + 120, False, "Time")  # span = 1320
    time_dim = routing.GetDimensionOrDie("Time")
    for i in range(1, num_locations):
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(0, tw_rel[1])
    for v in range(num_vehicles):
        idx = routing.Start(v)
        time_dim.CumulVar(idx).SetRange(0, tw_rel[1])
    for v in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # Solve
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.FromSeconds(30)

    solution = routing.SolveWithParameters(params)

    if not solution:
        return {"status": "infeasible"}

    # Extract routes
    routes = []
    total_distance = 0
    for v in range(num_vehicles):
        if not routing.IsVehicleUsed(solution, v):
            continue
        route = []
        idx = routing.Start(v)
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        routes.append(route)
        # Compute route distance
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]

    return {
        "status": "optimal",
        "num_routes": len(routes),
        "total_distance_m": total_distance,
        "routes": routes,
        "zone_ids": zone_ids,
    }

if __name__ == "__main__":
    cfg = load_config()
    forecast = pd.read_json("results/forecast.json")
    latest = forecast["date"].max()
    today = forecast[forecast["date"] == latest]
    result = solve_vrp(today, cfg)
    with open("results/routes.json", "w") as f:
        json.dump(result, f, indent=2)

    if result["status"] == "infeasible":
        print("No feasible solution found.")
        print(f"  Zones: {len(today)}, Capacity: {cfg['optimization']['vehicle_capacity']}, "
              f"Vehicles: {cfg['optimization']['num_vehicles']}")
        print(f"  Total demand: {today['forecast'].sum():.0f}")
    else:
        print(f"Routes: {result['num_routes']}, Total distance: {result['total_distance_m']} m")   