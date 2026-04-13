"""
astar_router.py — A* routing using LSTM-predicted FUTURE density as edge weights.

Key insight: edge weight uses predict_future_density(junction, ETA_at_that_junction)
not current density. The ambulance sees traffic as it will be when it arrives.

Depends on: junction_graph.py, lstm_model.py
"""

import heapq
import math

from junction_graph import (
    JUNCTIONS, HOSPITALS, junction_state,
    get_neighbors, euclidean, all_hospital_junctions,
)
from lstm_model import predict_future_density

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_TRAVEL_TIME_PER_JUNCTION = 30.0   # seconds (flat road, no congestion)
AMBULANCE_SPEED_MS             = 14.0  # m/s ≈ 50 km/h through city
SPILLBACK_PENALTY              = 45.0  # seconds added if junction is owned by another ambulance
MAX_DENSITY_DELAY              = 60.0  # max seconds added by congestion

# ---------------------------------------------------------------------------
# Edge weight
# ---------------------------------------------------------------------------

def _edge_weight(
    junction_id: str,
    eta_seconds: float,
    requesting_ambulance_id: str,
) -> float:
    """
    Compute travel-time cost of passing through junction_id when arriving at eta_seconds.

    Components:
      base_time       → BASE_TRAVEL_TIME_PER_JUNCTION
      density_penalty → LSTM-predicted density × MAX_DENSITY_DELAY
      queue_delay     → queue_size × 2s per vehicle
      spillback_pen   → if junction is held by a different ambulance corridor
    """
    base = BASE_TRAVEL_TIME_PER_JUNCTION

    predicted = predict_future_density(junction_id, eta_seconds)
    density_penalty = predicted * MAX_DENSITY_DELAY

    state = junction_state.get(junction_id, {})
    queue_delay = state.get("queue_size", 0) * 2.0

    corridor_owner = state.get("active_corridor")
    spillback = (
        SPILLBACK_PENALTY
        if (corridor_owner and corridor_owner != requesting_ambulance_id)
        else 0.0
    )

    return base + density_penalty + queue_delay + spillback


# ---------------------------------------------------------------------------
# A* implementation
# ---------------------------------------------------------------------------

def astar(
    start: str,
    goal: str,
    ambulance_id: str,
) -> tuple[list[str], float]:
    """
    Standard A* search from start to goal.

    g_score[node] tracks cumulative time in seconds (used as ETA for
    edge weight calculation to ensure future density is correctly timed).

    Returns (path, total_cost_seconds) or ([], inf) if unreachable.
    """
    # priority queue: (f_score, junction_id)
    open_heap: list[tuple[float, str]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[str, str] = {}
    g_score: dict[str, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return _reconstruct(came_from, current), g_score[current]

        for neighbor in get_neighbors(current):
            eta_at_neighbor = g_score[current]   # time when we reach the neighbor
            w = _edge_weight(neighbor, eta_at_neighbor, ambulance_id)
            tentative_g = g_score[current] + w

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = euclidean(neighbor, goal) / AMBULANCE_SPEED_MS
                heapq.heappush(open_heap, (tentative_g + h, neighbor))

    return [], math.inf


def _reconstruct(came_from: dict, current: str) -> list[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path


# ---------------------------------------------------------------------------
# Best hospital selection
# ---------------------------------------------------------------------------

def find_best_hospital_route(
    start: str,
    ambulance_id: str,
) -> tuple[list[str], str, str, float]:
    """
    Run A* to all 3 hospitals. Pick the lowest-cost one.

    Returns (route, hospital_id, hospital_name, travel_time_seconds).
    """
    best_route: list[str] = []
    best_hospital_id = ""
    best_hospital_name = ""
    best_time = math.inf

    for hid, hjunc, hname in all_hospital_junctions():
        if hjunc == start:
            # Already at the hospital junction — trivial path
            return [start], hid, hname, 0.0

        route, cost = astar(start, hjunc, ambulance_id)
        if cost < best_time:
            best_time = cost
            best_route = route
            best_hospital_id = hid
            best_hospital_name = hname

    if not best_route:
        # Absolute fallback: direct path via neighbors
        best_route = [start]
        best_hospital_id = "H1"
        best_hospital_name = HOSPITALS.get("H1", {}).get("name", "Hospital")
        best_time = 999.0

    return best_route, best_hospital_id, best_hospital_name, best_time


# ---------------------------------------------------------------------------
# Reroute check
# ---------------------------------------------------------------------------

def should_reroute(
    ambulance_id: str,
    current_junction: str,
    hospital_junction: str,
    current_route: list[str],
    improvement_threshold: float = 0.10,
) -> tuple[bool, list[str], float]:
    """
    Recompute A* from current_junction to hospital_junction.
    If new route is ≥ improvement_threshold (10%) faster, trigger reroute.

    Returns (should_reroute, new_route, new_time).
    """
    new_route, new_time = astar(current_junction, hospital_junction, ambulance_id)
    if not new_route:
        return False, current_route, 999.0

    # Estimate current remaining time naively
    remaining_junctions = max(len(current_route) - 1, 1)
    old_time_estimate = remaining_junctions * BASE_TRAVEL_TIME_PER_JUNCTION

    if old_time_estimate <= 0:
        return False, current_route, new_time

    improvement = (old_time_estimate - new_time) / old_time_estimate
    if improvement >= improvement_threshold:
        return True, new_route, new_time

    return False, current_route, new_time


# ---------------------------------------------------------------------------
# Turn-by-turn directions builder
# ---------------------------------------------------------------------------

def build_directions(route: list[str]) -> list[dict]:
    """
    Convert a junction list into turn-by-turn direction steps.

    Returns list of dicts: {junction, next_junction, turn, arrow, distance_m, road}
    """
    if len(route) < 2:
        return [{"junction": route[0] if route else "?",
                 "next_junction": None, "turn": "ARRIVED",
                 "arrow": "✓", "distance_m": 0, "road": "Destination"}]

    directions = []
    for i in range(len(route) - 1):
        j_cur  = route[i]
        j_next = route[i + 1]

        pos_cur  = JUNCTIONS[j_cur]["pos"]
        pos_next = JUNCTIONS[j_next]["pos"]

        dx = pos_next[0] - pos_cur[0]
        dy = pos_next[1] - pos_cur[1]   # y increases upward in the graph

        # Cardinal direction
        if abs(dy) >= abs(dx):
            turn = "STRAIGHT"
            arrow = "↑" if dy > 0 else "↓"
        elif dx > 0:
            turn = "RIGHT"
            arrow = "→"
        else:
            turn = "LEFT"
            arrow = "←"

        # Distance (Euclidean of grid coords, treated as metres)
        dist = ((dx ** 2) + (dy ** 2)) ** 0.5

        directions.append({
            "junction":      j_cur,
            "next_junction": j_next,
            "turn":          turn,
            "arrow":         arrow,
            "distance_m":    round(dist, 1),
            "road":          f"{j_cur} → {j_next}",
        })

    # Final step — arrival
    directions.append({
        "junction":      route[-1],
        "next_junction": None,
        "turn":          "ARRIVED",
        "arrow":         "✓",
        "distance_m":    0,
        "road":          "Hospital",
    })
    return directions
