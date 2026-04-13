"""
junction_graph.py — Static city junction graph + runtime state.
No external dependencies. All other modules import from here.
"""

# ---------------------------------------------------------------------------
# Static graph (matches existing map.net.xml junction ids J1–J17)
# ---------------------------------------------------------------------------
JUNCTIONS = {
    "J1":  {"pos": (0,   400), "neighbors": ["J2",  "J7"],              "lane_capacity": 20},
    "J2":  {"pos": (100, 400), "neighbors": ["J1",  "J3",  "J5"],       "lane_capacity": 24},
    "J3":  {"pos": (200, 400), "neighbors": ["J2",  "J4",  "J6"],       "lane_capacity": 18},
    "J4":  {"pos": (400, 400), "neighbors": ["J3",  "J11"],             "lane_capacity": 16},
    "J5":  {"pos": (100, 300), "neighbors": ["J2",  "J6",  "J8"],       "lane_capacity": 22},
    "J6":  {"pos": (200, 300), "neighbors": ["J5",  "J3",  "J9"],       "lane_capacity": 30},
    "J7":  {"pos": (0,   200), "neighbors": ["J1",  "J8",  "J14"],      "lane_capacity": 20},
    "J8":  {"pos": (100, 200), "neighbors": ["J5",  "J7",  "J9",  "J12"], "lane_capacity": 28},
    "J9":  {"pos": (200, 200), "neighbors": ["J6",  "J8",  "J10", "J13"], "lane_capacity": 26},
    "J10": {"pos": (300, 200), "neighbors": ["J9",  "J11", "J16"],      "lane_capacity": 24},
    "J11": {"pos": (400, 200), "neighbors": ["J10", "J4",  "J17"],      "lane_capacity": 18},
    "J12": {"pos": (100, 100), "neighbors": ["J8",  "J13", "J15"],      "lane_capacity": 22},
    "J13": {"pos": (200, 100), "neighbors": ["J12", "J9",  "J16"],      "lane_capacity": 20},
    "J14": {"pos": (0,   0),   "neighbors": ["J7",  "J15"],             "lane_capacity": 16},
    "J15": {"pos": (100, 0),   "neighbors": ["J14", "J12", "J16"],      "lane_capacity": 18},
    "J16": {"pos": (200, 0),   "neighbors": ["J15", "J13", "J10", "J17"], "lane_capacity": 24},
    "J17": {"pos": (400, 0),   "neighbors": ["J16", "J11"],             "lane_capacity": 14},
}

HOSPITALS = {
    "H1": {"junction": "J1",  "name": "City General Hospital"},
    "H2": {"junction": "J4",  "name": "East District Hospital"},
    "H3": {"junction": "J14", "name": "South Medical Center"},
}

# ---------------------------------------------------------------------------
# Runtime state — reset to defaults on startup.
# Mutated in-place by signal_controller, cv_pipeline, sumo_controller.
# ---------------------------------------------------------------------------
junction_state: dict[str, dict] = {
    jid: {
        "density_history":   [0.3] * 10,   # last 10 CV readings (0.0–1.0)
        "current_density":   0.3,
        "queue_size":        0,             # estimated vehicles queued cross-direction
        "signal_state":      "NORMAL",      # NORMAL | AMBULANCE_GREEN | SPILLBACK_LOCKED | DRAIN_OPEN
        "active_corridor":   None,          # ambulance_id that owns this junction, or None
        "status_label":      "NORMAL",      # for display in controller
        "signal_overridden": False,
    }
    for jid in JUNCTIONS
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_neighbors(junction_id: str) -> list[str]:
    return JUNCTIONS.get(junction_id, {}).get("neighbors", [])


def get_pos(junction_id: str) -> tuple[int, int]:
    return JUNCTIONS.get(junction_id, {}).get("pos", (0, 0))


def euclidean(j1: str, j2: str) -> float:
    x1, y1 = get_pos(j1)
    x2, y2 = get_pos(j2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def push_density(junction_id: str, density: float) -> None:
    """Append a new density reading, keep history capped at 10."""
    state = junction_state[junction_id]
    state["density_history"].append(density)
    if len(state["density_history"]) > 10:
        state["density_history"].pop(0)
    state["current_density"] = density
    # crude queue estimate: each 0.1 density unit ≈ 1 vehicle queued
    state["queue_size"] = int(density * state.get("lane_capacity",
                               JUNCTIONS[junction_id]["lane_capacity"]) * 0.4)


def get_hospital_junction(hospital_id: str) -> str:
    return HOSPITALS.get(hospital_id, {}).get("junction", "J1")


def all_hospital_junctions() -> list[tuple[str, str, str]]:
    """Returns [(hospital_id, junction_id, name), ...]"""
    return [(hid, hdata["junction"], hdata["name"])
            for hid, hdata in HOSPITALS.items()]
