"""Solve CVRP-TW using OR-Tools with forecasted demand as node weights."""

import json
import math

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.config import load_config
from src.optimization.baseline import nearest_neighbor


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Great-circle distance in meters between two lat/lon points."""
    R = 6_371_000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return int(R * 2 * math.asin(math.sqrt(a)))


def build_distance_matrix(zone_ids: list[int]) -> list[list[int]]:
    """Load zone centroids and compute pairwise haversine distances."""
    centroids = pd.read_csv("data/processed/zone_centroids.csv")
    centroids = centroids.set_index("PULocationID").loc[zone_ids].reset_index()

    lats = centroids["lat"].tolist()
    lons = centroids["lon"].tolist()
    n = len(zone_ids)

    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix


def solve_vrp(
    forecast: pd.DataFrame, cfg: dict, first_solution_only: bool = False
) -> dict:
    forecast = forecast.sort_values("PULocationID").reset_index(drop=True)
    zone_ids = forecast["PULocationID"].tolist()
    depot_zone = cfg["optimization"]["depot_zone"]
    if depot_zone not in zone_ids:
        return {
            "status": "error",
            "reason": f"depot zone {depot_zone} missing from forecast",
        }
    depot = zone_ids.index(depot_zone)

    demand = forecast["forecast"].clip(lower=0).round().astype(int).tolist()
    demand[depot] = 0
    print(
        f"  Total demand: {sum(demand)}, capacity/vehicle: {cfg['optimization']['vehicle_capacity']}"
    )

    num_locations = len(zone_ids)
    num_vehicles = cfg["optimization"]["num_vehicles"]
    capacity = cfg["optimization"]["vehicle_capacity"]
    tw = cfg["optimization"]["time_window"]

    dist_matrix = build_distance_matrix(zone_ids)
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def dist_cb(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)
        ]

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # Capacity dimension
    def demand_cb(from_index):
        return demand[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [capacity] * num_vehicles, True, "Capacity"
    )

    # Time window dimension
    def time_cb(from_index, to_index):
        # distance in meters / 30 km/h (8.33 m/s) → seconds → minutes
        return (
            dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
            // 500
        )

    time_idx = routing.RegisterTransitCallback(time_cb)
    tw_rel = [0, tw[1] - tw[0]]  # [0, 1440]

    routing.AddDimension(time_idx, 0, tw_rel[1] + 120, False, "Time")  # span = 1560
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
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if first_solution_only:
        params.solution_limit = 1  # baseline: no improvement search
    else:
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
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
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i + 1]]

    return {
        "status": "optimal",
        "num_routes": len(routes),
        "total_distance_m": total_distance,
        "routes": routes,
        "zone_ids": zone_ids,
    }


def prepare_instance(forecast: pd.DataFrame, cfg: dict):
    forecast = forecast.sort_values("PULocationID").reset_index(drop=True)
    zone_ids = forecast["PULocationID"].tolist()
    depot = zone_ids.index(cfg["optimization"]["depot_zone"])
    demand = forecast["forecast"].clip(lower=0).round().astype(int).tolist()
    demand[depot] = 0
    return zone_ids, depot, demand, build_distance_matrix(zone_ids)


if __name__ == "__main__":
    cfg = load_config()
    forecast = pd.read_json("results/forecast.json")
    latest = forecast["date"].max()
    today = forecast[forecast["date"] == latest]

    result = solve_vrp(today, cfg)  # full search
    first = solve_vrp(today, cfg, first_solution_only=True)  # no improvement

    zone_ids, depot, demand, dist = prepare_instance(today, cfg)
    nn = nearest_neighbor(
        dist,
        demand,
        cfg["optimization"]["vehicle_capacity"],
        depot,
        cfg["optimization"]["num_vehicles"],
    )

    if result.get("status") == "feasible" and not nn["unserved"]:
        opt, nn_d = result["total_distance_m"], nn["total_distance_m"]
        result["baselines"] = {
            "nearest_neighbor_m": nn_d,
            "first_solution_m": first.get("total_distance_m"),
            "reduction_vs_nn_pct": round((nn_d - opt) / nn_d * 100, 1),
        }

    with open("results/routes.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result.get("baselines", {}), indent=2))
