"""
sumo_controller.py — TraCI background thread + vehicle management.

Key coordinate design decision
--------------------------------
Vehicles are always emitted in one of two coordinate spaces:
  • coord_space = "graph"  → x,y are in junction-graph units (0–400 range).
                             Client maps them with toCanvas(x, y) — same as junctions.
  • coord_space = "sumo"   → x,y are raw SUMO metre coordinates.
                             Client maps them with sumoToCanvas(x, y) using net_boundary.

This eliminates every "tiny dash" bug caused by hardcoded canvas sizes.
"""

import os
import sys
import time
import math
import threading
from typing import Callable

from junction_graph import JUNCTIONS, HOSPITALS, junction_state, get_neighbors, euclidean

# ---------------------------------------------------------------------------
# SUMO home resolution
# ---------------------------------------------------------------------------

_CANDIDATE_SUMO_HOME = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mini project", "sumo_install", "sumo-1.19.0")
)

def _setup_sumo_home() -> str:
    if os.path.exists(_CANDIDATE_SUMO_HOME):
        os.environ["SUMO_HOME"] = _CANDIDATE_SUMO_HOME
        sys.path.append(os.path.join(_CANDIDATE_SUMO_HOME, "tools"))
        return _CANDIDATE_SUMO_HOME
    return os.environ.get("SUMO_HOME", "")


def _sumo_bin(name: str, sumo_home: str) -> str:
    for ext in (".exe", ""):
        candidate = os.path.join(sumo_home, "bin", name + ext) if sumo_home else name
        if os.path.exists(candidate):
            return candidate
    return name


# ---------------------------------------------------------------------------
# Graph coordinate helpers (used for fallback, 0-400 range)
# ---------------------------------------------------------------------------

_GX = [d["pos"][0] for d in JUNCTIONS.values()]
_GY = [d["pos"][1] for d in JUNCTIONS.values()]
_GXMIN, _GXMAX = min(_GX), max(_GX)
_GYMIN, _GYMAX = min(_GY), max(_GY)


def _interp_graph(j1: str, j2: str, t: float) -> tuple[float, float]:
    """Interpolate a position between two junction graph positions."""
    x1, y1 = JUNCTIONS[j1]["pos"]
    x2, y2 = JUNCTIONS[j2]["pos"]
    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t


def _compute_turn(cur_jid: str, nxt_jid: str | None) -> str:
    if nxt_jid is None:
        return "ARRIVED"
    cx, cy = JUNCTIONS[cur_jid]["pos"]
    nx, ny = JUNCTIONS[nxt_jid]["pos"]
    dx, dy = nx - cx, ny - cy
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "STRAIGHT"   # north/south movement


# ---------------------------------------------------------------------------
# SUMOController
# ---------------------------------------------------------------------------

class SUMOController:
    def __init__(self, socketio, cfg_path: str = "sumo_config/city.sumocfg", headless: bool = True):
        self.socketio  = socketio
        self.cfg_path  = os.path.abspath(cfg_path)
        self.headless  = headless
        self.running   = False
        self._thread: threading.Thread | None = None
        self._lock     = threading.Lock()

        # Fallback vehicle simulation
        self._fallback_vehicles: dict[str, dict] = {}

        # SUMO network boundary (set after traci.start)
        self._net_min_x  = 0.0
        self._net_min_y  = 0.0
        self._net_width  = 1000.0
        self._net_height = 1000.0

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> bool:
        sumo_home = _setup_sumo_home()
        binary = _sumo_bin("sumo" if self.headless else "sumo-gui", sumo_home)

        try:
            import traci
            traci.start([
                binary,
                "-c", self.cfg_path,
                "--start",
                "--no-step-log",
                "--no-warnings",
            ])
            print(f"[SUMO] Started {'headless' if self.headless else 'GUI'} simulation")

            # Read and broadcast net boundary so client can map raw SUMO coords
            try:
                boundary = traci.simulation.getNetBoundary()
                self._net_min_x  = boundary[0][0]
                self._net_min_y  = boundary[0][1]
                self._net_width  = max(boundary[1][0] - boundary[0][0], 1.0)
                self._net_height = max(boundary[1][1] - boundary[0][1], 1.0)
                self.socketio.emit("net_boundary", {
                    "minX": self._net_min_x, "minY": self._net_min_y,
                    "maxX": boundary[1][0],  "maxY": boundary[1][1],
                }, room="controller")
                print(f"[SUMO] Net boundary: {boundary}")
            except Exception as e:
                print(f"[SUMO] Could not read net boundary: {e}")

        except Exception as e:
            print(f"[SUMO] WARNING: Could not start SUMO ({e}). Running fallback mode.")
            self._start_fallback()
            return False

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sumo-thread")
        self._thread.start()
        return True

    # ------------------------------------------------------------------
    # Real SUMO loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main SUMO loop — 500 ms per tick.  Emits vehicles in 'sumo' coord space."""
        import traci
        from signal_controller import schedule_signal_clearance, restore_junction

        last_prox_check = 0.0

        while self.running:
            try:
                traci.simulationStep()
            except Exception as e:
                print(f"[SUMO] Sim step error: {e}")
                break

            # --- Collect vehicle data (raw SUMO coords) ---
            vehicles = []
            try:
                for vid in traci.vehicle.getIDList():
                    sx, sy  = traci.vehicle.getPosition(vid)
                    speed   = traci.vehicle.getSpeed(vid)
                    angle   = traci.vehicle.getAngle(vid)
                    vtype   = traci.vehicle.getTypeID(vid)
                    vehicles.append({
                        "id":          vid,
                        "x":           round(sx, 2),
                        "y":           round(sy, 2),
                        "angle":       round(angle, 1),
                        "speed":       round(speed * 3.6, 1),
                        "type":        "ambulance" if "ambulance" in vtype.lower() else "car",
                        "coord_space": "sumo",
                    })
            except Exception:
                pass

            # --- Signal states ---
            signals = {jid: s.get("signal_state", "NORMAL") for jid, s in junction_state.items()}

            # --- Push frame ---
            frame = {
                "vehicles":  vehicles,
                "signals":   signals,
                "junctions": {
                    jid: {
                        "signal":  signals[jid],
                        "density": round(junction_state[jid].get("current_density", 0), 2),
                    }
                    for jid in JUNCTIONS
                },
            }
            try:
                self.socketio.emit("sumo_frame", frame, room="controller")
            except Exception:
                pass

            # --- Ambulance proximity check every tick ---
            now = time.time()
            if now - last_prox_check > 0.5:
                last_prox_check = now
                self._check_ambulance_proximity(traci, schedule_signal_clearance, restore_junction)

            time.sleep(0.5)

        print("[SUMO] Loop ended.")

    # ------------------------------------------------------------------
    # Ambulance proximity tracking (SUMO mode)
    # ------------------------------------------------------------------

    def _check_ambulance_proximity(self, traci, schedule_fn, restore_fn) -> None:
        import ambulance_registry as _reg
        emit_fn = self.socketio.emit

        for amb in _reg.get_active():
            try:
                if amb.sumo_id not in traci.vehicle.getIDList():
                    continue

                sx, sy = traci.vehicle.getPosition(amb.sumo_id)
                speed  = traci.vehicle.getSpeed(amb.sumo_id)
                amb.sumo_position = (sx, sy)
                amb.speed_kmh     = round(speed * 3.6, 1)

                route = amb.route
                idx   = amb.current_junction_index

                # --- Check if ambulance has reached next junction ---
                if idx + 1 < len(route):
                    nxt_jid = route[idx + 1]
                    try:
                        jx, jy = traci.junction.getPosition(nxt_jid)
                    except Exception:
                        gx, gy = JUNCTIONS[nxt_jid]["pos"]
                        # Rough scale: graph units → SUMO metres (200 m grid spacing)
                        jx = gx * 200.0 / _GXMAX * self._net_width  + self._net_min_x
                        jy = gy * 200.0 / _GYMAX * self._net_height + self._net_min_y

                    dist_m = math.sqrt((sx - jx) ** 2 + (sy - jy) ** 2)
                    eta    = dist_m / max(speed, 0.5)

                    # Schedule signal clearance when within 60 s
                    if eta < 60 and nxt_jid in junction_state:
                        schedule_fn(nxt_jid, eta, amb.id, emit_fn)

                    # Advance when within 40 m
                    if dist_m < 40:
                        restore_fn(route[idx], emit_fn)
                        amb.current_junction_index = min(idx + 1, len(route) - 1)
                        idx = amb.current_junction_index
                        cur_j  = route[idx]
                        nxt_j  = route[idx + 1] if idx + 1 < len(route) else None
                        turn   = _compute_turn(cur_j, nxt_j)
                        jleft  = len(route) - idx - 1
                        dist_r = jleft * 200
                        eta_s  = jleft * 30

                        try:
                            self.socketio.emit("position_update", {
                                "current_junction":   cur_j,
                                "next_junction":      nxt_j,
                                "next_turn":          turn,
                                "distance_remaining": dist_r,
                                "eta_seconds":        eta_s,
                                "speed_kmh":          amb.speed_kmh,
                            }, room=amb.id)
                            self.socketio.emit("ambulance_updated", amb.to_summary(), room="controller")
                        except Exception:
                            pass

                # --- Arrival check ---
                remaining = len(route) - 1 - amb.current_junction_index
                if remaining <= 0 and amb.status != "ARRIVED":
                    amb.status       = "ARRIVED"
                    amb.arrival_time = time.time()
                    amb.time_saved   = max(0.0, amb.estimated_travel_time - (amb.arrival_time - amb.dispatch_time))
                    try:
                        traci.vehicle.remove(amb.sumo_id)
                    except Exception:
                        pass
                    self._on_ambulance_arrived(amb)

            except Exception:
                pass

    def _on_ambulance_arrived(self, amb) -> None:
        from signal_controller import restore_all_for_ambulance
        restore_all_for_ambulance(amb.id, self.socketio.emit)
        ts = time.strftime("%H:%M:%S")
        time_taken = round(time.time() - amb.dispatch_time, 1)
        try:
            self.socketio.emit("ambulance_arrived", {
                "id":                amb.id,
                "hospital":          amb.hospital_name,
                "time_taken":        time_taken,
                "time_saved":        round(amb.time_saved, 1),
                "signals_overridden": amb.signals_overridden,
            }, room="controller")
            self.socketio.emit("arrived", {
                "hospital_name": amb.hospital_name,
                "time_taken":    time_taken,
                "time_saved":    round(amb.time_saved, 1),
            }, room=amb.id)
            self.socketio.emit("event_log", {
                "timestamp": ts,
                "message":   f"🏥 {amb.id} ARRIVED at {amb.hospital_name} — saved {amb.time_saved:.0f}s",
                "level":     "INFO",
            }, room="controller")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # add_ambulance — exhaustive edge detection
    # ------------------------------------------------------------------

    def add_ambulance(self, ambulance_id: str, route: list[str]) -> None:
        """Inject an ambulance into SUMO with exhaustive edge matching."""
        if not self.running:
            self._add_fallback_vehicle(ambulance_id, route)
            return
        try:
            import traci
            all_edges = set(traci.edge.getIDList())

            # Build per-pair edge list
            edges = []
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                found_edge = None

                # Attempt many naming patterns
                candidates = [
                    f"{a}_to_{b}", f"{b}_to_{a}",
                    f"{a}to{b}",   f"{b}to{a}",
                    f"{a}_{b}",    f"{b}_{a}",
                    f"{a}-{b}",    f"{b}-{a}",
                    f"{a}{b}",     f"{b}{a}",
                    a.lower() + "to" + b.lower(),
                    b.lower() + "to" + a.lower(),
                ]
                for c in candidates:
                    if c in all_edges:
                        found_edge = c
                        break

                # Last resort: iterate edges by from/to node
                if not found_edge:
                    for eid in all_edges:
                        try:
                            if (traci.edge.getFromJunction(eid) == a and
                                    traci.edge.getToJunction(eid) == b):
                                found_edge = eid
                                break
                        except Exception:
                            try:
                                if (traci.edge.getFromNode(eid) == a and
                                        traci.edge.getToNode(eid) == b):
                                    found_edge = eid
                                    break
                            except Exception:
                                pass

                if found_edge:
                    edges.append(found_edge)
                else:
                    print(f"[SUMO] No edge found {a}→{b} (edges sample: {list(all_edges)[:8]})")

            if not edges:
                print(f"[SUMO] No valid edges for {ambulance_id}; using fallback")
                self._add_fallback_vehicle(ambulance_id, route)
                return

            # Ensure ambulance vehicle type exists
            try:
                existing_types = traci.vehicletype.getIDList()
                if "ambulance" not in existing_types:
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance")
                traci.vehicletype.setColor("ambulance", (255, 255, 255, 255))
                traci.vehicletype.setMaxSpeed("ambulance", 22.22)
            except Exception as e:
                print(f"[SUMO] vtype warning: {e}")

            route_id = f"route_{ambulance_id}"
            try:
                if route_id in traci.route.getIDList():
                    pass  # already registered
                else:
                    traci.route.add(route_id, edges)
            except Exception as e:
                print(f"[SUMO] route.add warning: {e}")

            try:
                traci.vehicle.add(
                    ambulance_id, route_id,
                    typeID="ambulance",
                    depart="now",
                    departLane="best",
                    departSpeed="max",
                )
                traci.vehicle.setColor(ambulance_id, (255, 255, 255, 255))
                traci.vehicle.setSpeedMode(ambulance_id, 0)
                traci.vehicle.setLaneChangeMode(ambulance_id, 0)
                print(f"[SUMO] Ambulance {ambulance_id} added on {len(edges)} edges")
            except Exception as e:
                print(f"[SUMO] vehicle.add failed: {e}")
                self._add_fallback_vehicle(ambulance_id, route)

        except Exception as e:
            print(f"[SUMO] add_ambulance error: {e}")
            self._add_fallback_vehicle(ambulance_id, route)

    # ------------------------------------------------------------------
    # Fallback simulation (no SUMO)
    # ------------------------------------------------------------------

    def _start_fallback(self) -> None:
        self.running = False
        self._thread = threading.Thread(
            target=self._fallback_loop, daemon=True, name="fallback-sumo"
        )
        self._thread.start()
        print("[SUMO] Running in Python fallback mode (no SUMO)")

    def _add_fallback_vehicle(self, ambulance_id: str, route: list[str]) -> None:
        self._fallback_vehicles[ambulance_id] = {
            "route":    route,
            "step":     0,
            "progress": 0.0,   # 0.0–1.0 within current segment
        }

    def _fallback_loop(self) -> None:
        """
        Move fallback ambulances every 0.5 s.
        Vehicles emitted in GRAPH coordinate space (0–400) so the client's
        existing toCanvas() function maps them identically to junction circles.
        """
        import ambulance_registry as _reg
        from signal_controller import restore_all_for_ambulance

        # Each junction step takes TICKS_PER_JUNCTION × 0.5 s ≈ 8 s
        TICKS_PER_JUNCTION = 16

        while True:
            signals  = {jid: s.get("signal_state", "NORMAL") for jid, s in junction_state.items()}
            vehicles = []

            for vid, fv in list(self._fallback_vehicles.items()):
                route = fv["route"]
                step  = fv["step"]

                if step >= len(route) - 1:
                    # Parked at destination
                    gx, gy = JUNCTIONS[route[-1]]["pos"]
                    vehicles.append({
                        "id": vid, "x": gx, "y": gy,
                        "angle": 0, "speed": 0, "type": "ambulance",
                        "coord_space": "graph",
                    })
                    continue

                # Advance progress
                fv["progress"] = fv.get("progress", 0.0) + (1.0 / TICKS_PER_JUNCTION)

                if fv["progress"] >= 1.0:
                    fv["progress"] = 0.0
                    new_step = min(step + 1, len(route) - 1)
                    fv["step"] = new_step

                    # Sync ambulance state
                    try:
                        amb = _reg.get(vid)
                        if amb and amb.status != "ARRIVED":
                            amb.current_junction_index = new_step
                            amb.estimated_travel_time  = max(0, amb.estimated_travel_time - 8)

                            if new_step >= len(route) - 1:
                                # === ARRIVED ===
                                amb.status       = "ARRIVED"
                                amb.arrival_time = time.time()
                                amb.time_saved   = max(0.0, amb.estimated_travel_time)
                                restore_all_for_ambulance(vid, None)
                                ts = time.strftime("%H:%M:%S")
                                time_taken = round(time.time() - amb.dispatch_time, 1)
                                try:
                                    self.socketio.emit("ambulance_arrived", {
                                        "id":               vid,
                                        "hospital":         amb.hospital_name,
                                        "time_taken":       time_taken,
                                        "time_saved":       round(amb.time_saved, 1),
                                        "signals_overridden": amb.signals_overridden,
                                    }, room="controller")
                                    self.socketio.emit("arrived", {
                                        "hospital_name": amb.hospital_name,
                                        "time_taken":    time_taken,
                                        "time_saved":    round(amb.time_saved, 1),
                                    }, room=vid)
                                    self.socketio.emit("event_log", {
                                        "timestamp": ts,
                                        "message":   f"🏥 {vid} ARRIVED at {amb.hospital_name} — saved {round(amb.time_saved)}s",
                                        "level":     "INFO",
                                    }, room="controller")
                                except Exception:
                                    pass
                            else:
                                # === EN ROUTE — broadcast position update ===
                                cur_j = route[new_step]
                                nxt_j = route[new_step + 1] if new_step + 1 < len(route) else None
                                turn  = _compute_turn(cur_j, nxt_j)
                                jleft = len(route) - new_step - 1
                                eta_s = max(0, amb.estimated_travel_time)

                                try:
                                    self.socketio.emit("position_update", {
                                        "current_junction":   cur_j,
                                        "next_junction":      nxt_j,
                                        "next_turn":          turn,
                                        "distance_remaining": jleft * 200,
                                        "eta_seconds":        eta_s,
                                        "speed_kmh":          50.0,
                                    }, room=vid)
                                    self.socketio.emit("ambulance_updated",
                                                       amb.to_summary(), room="controller")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Interpolate in GRAPH coordinate space
                cur_step = fv["step"]
                j1 = route[cur_step]
                j2 = route[min(cur_step + 1, len(route) - 1)]
                gx, gy = _interp_graph(j1, j2, fv["progress"])

                vehicles.append({
                    "id": vid, "x": round(gx, 2), "y": round(gy, 2),
                    "angle": 0, "speed": 50, "type": "ambulance",
                    "coord_space": "graph",
                })

            # Build junction info (no pixel coords — client uses graph pos natively)
            junc_info = {
                jid: {
                    "signal":  signals[jid],
                    "density": round(junction_state[jid].get("current_density", 0), 2),
                }
                for jid in JUNCTIONS
            }

            frame = {"vehicles": vehicles, "signals": signals, "junctions": junc_info}
            try:
                self.socketio.emit("sumo_frame", frame, room="controller")
            except Exception:
                pass

            time.sleep(0.5)

    def stop(self) -> None:
        self.running = False
        try:
            import traci
            traci.close()
        except Exception:
            pass
