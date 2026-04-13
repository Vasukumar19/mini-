"""
app.py — Flask + SocketIO server for the Intelligent Ambulance Traffic System.

Routes:
  GET /controller  → controller.html (supervisor view)
  GET /driver      → driver.html     (individual driver tab, ?id=ambu_01)

WebSocket rooms:
  "controller"  → all ambulances aggregated, SUMO canvas frames
  "ambu_01" etc → each driver tab isolated to its own room

Start:
  python app.py
"""

import os
import sys
import time
import threading

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, emit

# ---------------------------------------------------------------------------
# Resolve SUMO path before any traci imports happen in sub-modules
# ---------------------------------------------------------------------------
_SUMO_CANDIDATE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mini project", "sumo_install", "sumo-1.19.0")
)
if os.path.exists(_SUMO_CANDIDATE):
    os.environ["SUMO_HOME"] = _SUMO_CANDIDATE
    sys.path.append(os.path.join(_SUMO_CANDIDATE, "tools"))

# ---------------------------------------------------------------------------
# Application imports (after SUMO path is set)
# ---------------------------------------------------------------------------
from junction_graph import JUNCTIONS, HOSPITALS, junction_state, all_hospital_junctions
from ambulance_state import AmbulanceState
import ambulance_registry as registry
from astar_router import find_best_hospital_route, build_directions, should_reroute
from signal_controller import (
    schedule_signal_clearance, restore_all_for_ambulance, cancel_scheduled_signals, init as sc_init
)
from conflict_resolver import check_for_conflicts
from sumo_controller import SUMOController
from cv_pipeline import start_cv_pipeline

# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "ambulance-traffic-system-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",   # use native Python threads (works without eventlet)
    logger=False,
    engineio_logger=False,
)

# ---------------------------------------------------------------------------
# Global subsystem instances
# ---------------------------------------------------------------------------

_sumo: SUMOController | None = None
_cv_stop_event = None
_ambulance_counter = 1    # auto-increment for "Open Driver" button
_conflict_check_stop = threading.Event()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("controller.html")


@app.route("/controller")
def controller():
    return render_template("controller.html")


@app.route("/driver")
def driver():
    driver_id = request.args.get("id", "ambu_01")
    return render_template("driver.html", driver_id=driver_id)


@app.route("/api/junctions")
def api_junctions():
    """Serve static junction graph for canvas drawing."""
    data = {
        jid: {
            "pos":      jdata["pos"],
            "neighbors": jdata["neighbors"],
        }
        for jid, jdata in JUNCTIONS.items()
    }
    hospitals_out = {
        hid: {"junction": hdata["junction"], "name": hdata["name"]}
        for hid, hdata in HOSPITALS.items()
    }
    return jsonify({"junctions": data, "hospitals": hospitals_out})


@app.route("/api/status")
def api_status():
    return jsonify({
        "ambulances": registry.summary_list(),
        "junction_states": {
            jid: {
                "signal_state":   s.get("signal_state", "NORMAL"),
                "current_density": round(s.get("current_density", 0), 3),
                "status_label":   s.get("status_label", "NORMAL"),
            }
            for jid, s in junction_state.items()
        },
    })


@app.route("/debug/edges")
def debug_edges():
    """Debug endpoint — returns SUMO edge and node IDs for edge-name troubleshooting.
    Visit http://localhost:5000/debug/edges after startup.
    """
    try:
        import traci
        edges = list(traci.edge.getIDList())[:50]
        nodes = list(traci.node.getIDList()) if hasattr(traci, 'node') else []
        # Sample from/to for first 10 edges
        sample = {}
        for eid in edges[:10]:
            try:
                sample[eid] = {
                    "from": traci.edge.getFromJunction(eid),
                    "to":   traci.edge.getToJunction(eid),
                }
            except Exception:
                try:
                    sample[eid] = {"from": traci.edge.getFromNode(eid), "to": traci.edge.getToNode(eid)}
                except Exception:
                    sample[eid] = {}
        return jsonify({"edges": edges, "nodes": nodes[:20], "sample_from_to": sample})
    except Exception as e:
        return jsonify({"error": str(e), "note": "SUMO not running — edges unavailable"})


# ---------------------------------------------------------------------------
# SocketIO events — connection management
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    pass   # no auto-room assignment; let subsequent events handle it


@socketio.on("register_driver")
def on_register_driver(data):
    """Driver tab registers with its unique driver_id."""
    driver_id = data.get("driver_id", "ambu_00")
    join_room(driver_id)
    emit("registered", {"driver_id": driver_id, "status": "ok"})
    _log(f"Driver tab connected: {driver_id}", "INFO")


@socketio.on("join_controller")
def on_join_controller(data=None):
    """Supervisor joins the controller room to receive all events."""
    join_room("controller")
    # Send current ambulance list immediately
    emit("ambulance_list", {"ambulances": registry.summary_list()})
    # Send junction graph
    emit("junction_graph", {
        "junctions": {jid: {"pos": v["pos"], "neighbors": v["neighbors"]}
                      for jid, v in JUNCTIONS.items()},
        "hospitals":  {hid: {"junction": v["junction"], "name": v["name"]}
                       for hid, v in HOSPITALS.items()},
    })
    _log("Controller connected", "INFO")


# ---------------------------------------------------------------------------
# SocketIO events — dispatch
# ---------------------------------------------------------------------------

@socketio.on("dispatch")
def on_dispatch(data):
    """
    Driver tab dispatches an ambulance.
    Payload: {driver_id, junction, severity, patients}
    """
    driver_id = data.get("driver_id", "ambu_01")
    junction  = data.get("junction", "J5")
    severity  = data.get("severity", "HIGH")
    patients  = int(data.get("patients", 1))
    ts        = time.strftime("%H:%M:%S")

    _log(f"{driver_id} dispatched from {junction} ({severity}, {patients} patients)", "INFO")

    # Create state object
    amb = AmbulanceState(
        id=driver_id,
        origin_junction=junction,
        severity=severity,
        patient_count=patients,
    )
    registry.register(amb)

    # A* routing to best hospital
    try:
        route, h_id, h_name, travel_time = find_best_hospital_route(junction, driver_id)
    except Exception as e:
        _log(f"Routing error for {driver_id}: {e}", "WARN")
        route       = [junction, junction]   # trivial fallback
        h_id        = "H1"
        h_name      = "City General Hospital"
        travel_time = 999.0

    amb.route                  = route
    amb.hospital_id            = h_id
    amb.hospital_name          = h_name
    amb.estimated_travel_time  = travel_time
    amb.status                 = "EN_ROUTE"

    directions = build_directions(route)

    # Schedule signal corridor for entire route
    for i, junc_id in enumerate(route):
        eta = i * 30.0   # 30 s per junction estimate
        socketio.emit("signal_update", {
            "junction": junc_id,
            "status":   "PREPARING",
            "color":    "#ffaa00",
        }, room=driver_id)
        schedule_signal_clearance(junc_id, eta, driver_id, socketio.emit)

    # Add ambulance to SUMO
    if _sumo:
        _sumo.add_ambulance(driver_id, route)

    # Emit route to driver's private room
    socketio.emit("route_assigned", {
        "route":        route,
        "hospital_id":  h_id,
        "hospital_name": h_name,
        "eta_seconds":  round(travel_time, 1),
        "directions":   directions,
    }, room=driver_id)

    # Notify controller about new ambulance
    socketio.emit("ambulance_added", {
        "id":           driver_id,
        "severity":     severity,
        "patients":     patients,
        "route":        route,
        "hospital":     h_name,
        "eta":          round(travel_time, 1),
    }, room="controller")

    socketio.emit("event_log", {
        "timestamp": ts,
        "message":   f"🚑 {driver_id} dispatched from {junction} → {h_name}",
        "level":     "INFO",
    }, room="controller")

    _log(f"Route assigned to {driver_id}: {route} → {h_name} ({travel_time:.0f}s)", "INFO")


# ---------------------------------------------------------------------------
# SocketIO events — reroute
# ---------------------------------------------------------------------------

@socketio.on("reroute_check")
def on_reroute_check(data):
    """
    Called every 15 s per driver tab while en route.
    Payload: {driver_id, current_junction}
    """
    driver_id       = data.get("driver_id")
    current_junction = data.get("current_junction")
    if not driver_id or not current_junction:
        return

    amb = registry.get(driver_id)
    if not amb or amb.status == "ARRIVED":
        return

    # Find hospital junction from hospital_id
    h_junction = None
    for hid, hdata in HOSPITALS.items():
        if hid == amb.hospital_id:
            h_junction = hdata["junction"]
            break
    if not h_junction:
        return

    trigger, new_route, new_time = should_reroute(
        driver_id, current_junction, h_junction, amb.route
    )

    if trigger:
        old_eta = amb.estimated_travel_time
        amb.route                 = new_route
        amb.estimated_travel_time = new_time
        amb.reroute_count        += 1
        amb.time_saved           += max(0, old_eta - new_time)

        directions = build_directions(new_route)

        # Reschedule signals for new route
        cancel_scheduled_signals(amb.route)
        for i, jid in enumerate(new_route):
            schedule_signal_clearance(jid, i * 30.0, driver_id, socketio.emit)

        if _sumo and _sumo.running:
            _sumo.add_ambulance(driver_id, new_route)

        ts = time.strftime("%H:%M:%S")
        time_saved = round(old_eta - new_time, 1)

        socketio.emit("reroute", {
            "new_route":  new_route,
            "old_eta":    round(old_eta, 1),
            "new_eta":    round(new_time, 1),
            "time_saved": time_saved,
            "directions": directions,
        }, room=driver_id)

        socketio.emit("event_log", {
            "timestamp": ts,
            "message":   f"🔄 REROUTED {driver_id} — saves {time_saved:.0f}s",
            "level":     "INFO",
        }, room="controller")

        socketio.emit("ambulance_updated", amb.to_summary(), room="controller")


# ---------------------------------------------------------------------------
# SocketIO events — simulation speed control
# ---------------------------------------------------------------------------

@socketio.on("set_speed")
def on_set_speed(data):
    speed = float(data.get("speed", 1.0))   # 0.5 | 1.0 | 2.0 | 3.0
    try:
        import traci
        traci.simulation.setDeltaT(0.5 / speed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Background: conflict detection loop
# ---------------------------------------------------------------------------

def _conflict_loop() -> None:
    """Every 2 s, check for junction conflicts between active ambulances."""
    while not _conflict_check_stop.is_set():
        active = registry.get_active()
        if len(active) >= 2:
            conflicts = check_for_conflicts(active)
            for c in conflicts:
                ts = time.strftime("%H:%M:%S")
                # Slow down the loser via TraCI
                loser_id = c["loser"]
                try:
                    import traci
                    loser_amb = registry.get(loser_id)
                    if loser_amb and loser_amb.sumo_id:
                        traci.vehicle.setSpeed(loser_amb.sumo_id, 0)
                        threading.Timer(
                            20.0,
                            lambda sid=loser_amb.sumo_id: _resume_vehicle(sid)
                        ).start()
                except Exception:
                    pass

                # Emit to controller
                payload = dict(c)
                payload.pop("winner_obj", None)
                payload.pop("loser_obj", None)
                socketio.emit("conflict_detected", payload, room="controller")
                socketio.emit("event_log", {
                    "timestamp": ts,
                    "message":   (f"⚠️ CONFLICT: {c['winner']} vs {c['loser']} at {c['junction']} "
                                  f"— {c['winner']} wins ({c['winner_score']} vs {c['loser_score']})"),
                    "level":     "ALERT",
                }, room="controller")

                # Track metrics
                winner_amb = registry.get(c["winner"])
                loser_amb  = registry.get(c["loser"])
                if winner_amb: winner_amb.conflicts_resolved += 1
                if loser_amb:  loser_amb.conflicts_resolved  += 1

        _conflict_check_stop.wait(2.0)


def _resume_vehicle(sumo_id: str) -> None:
    try:
        import traci
        traci.vehicle.setSpeed(sumo_id, -1)   # restore normal speed control
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Background: periodic ambulance state push
# ---------------------------------------------------------------------------

def _state_push_loop() -> None:
    """Every 1.5 s push updated ambulance summaries to the controller."""
    while not _conflict_check_stop.is_set():
        ambs = registry.get_active()
        for amb in ambs:
            socketio.emit("ambulance_updated", amb.to_summary(), room="controller")
        _conflict_check_stop.wait(1.5)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[App] {ts} {msg}")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup() -> None:
    global _sumo, _cv_stop_event

    # 1. Generate SUMO network files if needed
    cfg = os.path.join(os.path.dirname(__file__), "sumo_config", "city.sumocfg")
    if not os.path.exists(cfg):
        print("[App] SUMO config missing — running generate_sumo_network.py …")
        import subprocess
        subprocess.run([sys.executable, "generate_sumo_network.py"])

    # 2. Train LSTM if weights missing
    model_path = os.path.join(os.path.dirname(__file__), "models", "lstm_traffic.pth")
    if not os.path.exists(model_path):
        print("[App] LSTM weights missing — running lstm_train.py …")
        from lstm_train import train
        train(epochs=50, n_days=30)   # quick 30-day training on startup

    # 3. Initialize signal controller with socketio
    sc_init(socketio, socketio.emit)

    # 4. Start CV pipeline background thread
    _cv_stop_event = start_cv_pipeline(socketio)

    # 5. Start SUMO controller
    _sumo = SUMOController(socketio, cfg_path=cfg, headless=True)
    _sumo.start()

    # 6. Start conflict detection loop
    t_conflict = threading.Thread(target=_conflict_loop, daemon=True, name="conflict-loop")
    t_conflict.start()

    # 7. Start state push loop
    t_state = threading.Thread(target=_state_push_loop, daemon=True, name="state-push")
    t_state.start()

    print("[App] ✅ All subsystems started. Open http://localhost:5000/controller")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _startup()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
