"""Quick integration test — run before starting app.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

SUMO_CANDIDATE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mini project", "sumo_install", "sumo-1.19.0"))
if os.path.exists(SUMO_CANDIDATE):
    os.environ["SUMO_HOME"] = SUMO_CANDIDATE

# ── Backend imports ──────────────────────────────────────────────────────────
from junction_graph import JUNCTIONS, HOSPITALS, junction_state
from ambulance_state import AmbulanceState
import ambulance_registry as registry
from astar_router import find_best_hospital_route, build_directions, should_reroute
from signal_controller import schedule_signal_clearance, restore_all_for_ambulance
from conflict_resolver import check_for_conflicts, resolve_conflict
print("All backend imports: OK")

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask
from flask_socketio import SocketIO
print("Flask + SocketIO imports: OK")

# ── End-to-end dispatch ──────────────────────────────────────────────────────
amb = AmbulanceState(id="test_01", origin_junction="J8", severity="HIGH", patient_count=3)
registry.register(amb)
route, hid, hname, eta = find_best_hospital_route("J8", "test_01")
dirs = build_directions(route)
amb.route = route
amb.hospital_id = hid
amb.hospital_name = hname
amb.estimated_travel_time = eta
print(f"Dispatch test: {amb.id} -> {hname} via {route} ({eta:.0f}s)")
print(f"Directions: {len(dirs)} steps, first turn: {dirs[0]['turn']}")

# ── Conflict detection ───────────────────────────────────────────────────────
amb2 = AmbulanceState(id="test_02", origin_junction="J9", severity="MEDIUM", patient_count=1)
amb2.route = ["J9", "J8", "J1"]
registry.register(amb2)
conflicts = check_for_conflicts([amb, amb2])
print(f"Conflict test: {len(conflicts)} conflicts detected (expected 0 unless same next_junction)")

# ── LSTM prediction ──────────────────────────────────────────────────────────
from lstm_model import predict_future_density
densities = [(j, predict_future_density(j, 60)) for j in ["J1","J5","J9","J14","J17"]]
print("LSTM predictions (60s ahead):")
for jid, d in densities:
    bar = "#" * int(d * 20)
    print(f"  {jid}: {d:.3f} [{bar:<20}]")

# ── routing with multiple ambulances ────────────────────────────────────────
print()
print("Multi-ambulance routing test:")
for origin, severity in [("J1","HIGH"), ("J8","MEDIUM"), ("J16","LOW")]:
    aid = f"ambu_test_{origin}"
    r, hid2, hname2, t = find_best_hospital_route(origin, aid)
    print(f"  {origin} ({severity}) -> {hname2}: {r} [{t:.0f}s]")

print()
print("=" * 40)
print("  FULL INTEGRATION TEST PASSED")
print("=" * 40)
print()
print("Next step: python app.py")
print("  Controller: http://localhost:5000/controller")
print("  Driver tab: http://localhost:5000/driver?id=ambu_01")
