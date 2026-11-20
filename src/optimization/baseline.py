def nearest_neighbor(dist_matrix, demand, capacity, depot, max_vehicles):
    unvisited = {i for i in range(len(demand)) if i != depot}
    routes, total = [], 0
    while unvisited and len(routes) < max_vehicles:
        route, load, cur = [depot], 0, depot
        while True:
            cands = [j for j in unvisited if load + demand[j] <= capacity]
            if not cands:
                break
            nxt = min(cands, key=lambda j: dist_matrix[cur][j])
            route.append(nxt); load += demand[nxt]; total += dist_matrix[cur][nxt]
            unvisited.remove(nxt); cur = nxt
        route.append(depot); total += dist_matrix[cur][depot]
        routes.append(route)
    return {"routes": routes, "total_distance_m": total, "unserved": sorted(unvisited)}


