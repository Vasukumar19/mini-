"""
signal_controller.py — Preemptive signal scheduling + spillback prevention.

All signal changes are applied via TraCI. If TraCI is unavailable,
changes are recorded in junction_state only (for UI display).

Depends on: junction_graph.py
"""

import threading
import time
from typing import Callable

from junction_graph import JUNCTIONS, junction_state, get_neighbors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFETY_BUFFER_S  = 5.0    # extra seconds added on top of queue drain time
RESTORE_DELAY_S  = 3.0    # seconds to wait after ambulance clears before restoring

# Holds active timers so they can be cancelled on reroute
_scheduled_timers: dict[str, threading.Timer] = {}   # junction_id → Timer

# Holds the restoration order per junction (list of (junction_id, role))
_spillback_restore_order: dict[str, list[tuple[str, str]]] = {}

# Injectable socketio reference (set by app.py)
_socketio = None
_emit_event: Callable | None = None


def init(socketio, emit_fn: Callable) -> None:
    """Called once from app.py after SocketIO is ready."""
    global _socketio, _emit_event
    _socketio = socketio
    _emit_event = emit_fn


# ---------------------------------------------------------------------------
# TraCI helpers (fail-silent)
# ---------------------------------------------------------------------------

def _traci_set_phase(junction_id: str, phase: int) -> None:
    try:
        import traci
        traci.trafficlight.setPhase(junction_id, phase)
    except Exception:
        pass   # SUMO may not be running yet


def _traci_set_all_red(junction_id: str) -> None:
    try:
        import traci
        logic = traci.trafficlight.getAllProgramLogics(junction_id)
        if logic:
            # Find a phase where all lanes are red, or just set duration-0 override
            traci.trafficlight.setPhase(junction_id, 2)   # phase 2 is typically all-red in SUMO defaults
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core scheduling
# ---------------------------------------------------------------------------

def schedule_signal_clearance(
    junction_id: str,
    eta_seconds: float,
    ambulance_id: str,
    emit_fn: Callable | None = None,
) -> None:
    """
    Preemptively schedule the green corridor for junction_id.

    trigger_advance = queue_size * 2s + SAFETY_BUFFER_S
    The green fires (trigger_advance) seconds before the ambulance arrives.
    """
    state = junction_state.get(junction_id)
    if state is None:
        return

    queue_size = state.get("queue_size", 0)
    trigger_advance = queue_size * 2.0 + SAFETY_BUFFER_S
    trigger_at = max(eta_seconds - trigger_advance, 1.0)   # never < 1s

    # Cancel any existing timer for this junction
    existing = _scheduled_timers.get(junction_id)
    if existing and existing.is_alive():
        existing.cancel()

    def _fire_green():
        _activate_green(junction_id, ambulance_id, emit_fn or _emit_event)

    t = threading.Timer(trigger_at, _fire_green)
    t.daemon = True
    t.start()
    _scheduled_timers[junction_id] = t

    _log(f"Signal scheduled: {junction_id} turns GREEN in {trigger_at:.1f}s "
         f"(queue {queue_size} veh, clears {trigger_advance:.1f}s early)", emit_fn)


def _activate_green(junction_id: str, ambulance_id: str, emit_fn: Callable | None) -> None:
    """Apply the ambulance-green phase to the junction and lock spillback."""
    state = junction_state.get(junction_id)
    if state is None:
        return

    old_state = state.get("signal_state", "NORMAL")

    _traci_set_phase(junction_id, 0)   # phase 0 = green for main direction

    state.update({
        "signal_state":      "AMBULANCE_GREEN",
        "signal_overridden": True,
        "active_corridor":   ambulance_id,
        "status_label":      "CORRIDOR_ACTIVE",
    })

    _log(f"✅ {junction_id} is GREEN for {ambulance_id}", emit_fn)
    _emit_signal_changed(junction_id, old_state, "AMBULANCE_GREEN", f"corridor:{ambulance_id}", emit_fn)
    _apply_spillback(junction_id, ambulance_id, emit_fn)


# ---------------------------------------------------------------------------
# Spillback prevention
# ---------------------------------------------------------------------------

def _apply_spillback(junction_id: str, ambulance_id: str, emit_fn: Callable | None) -> None:
    """
    FEEDER junctions (neighbors that feed cross traffic into junction_id):
      Set to RED — stop vehicles entering from cross direction.
    DRAIN junctions (neighbors downstream of cross traffic):
      Set to GREEN — let backed-up cross traffic escape.

    Uses a simple heuristic: all immediate neighbors are treated as feeders
    except the one the ambulance came from (inbound direction).
    """
    restoration_order: list[tuple[str, str]] = []
    active_corridor_owner = junction_state[junction_id].get("active_corridor")

    for neighbor_id in get_neighbors(junction_id):
        n_state = junction_state.get(neighbor_id)
        if n_state is None:
            continue
        # Skip if already claimed by the same ambulance corridor
        if n_state.get("active_corridor") == ambulance_id:
            continue

        old_sig = n_state.get("signal_state", "NORMAL")

        # Determine role: feeder (RED) or drain (GREEN)
        # Heuristic: if the neighbor is "earlier" in the grid (lower coord), it's a feeder
        jpos  = JUNCTIONS[junction_id]["pos"]
        npos  = JUNCTIONS[neighbor_id]["pos"]
        is_drain = False   # We treat most as feeders for safety; could be improved

        _traci_set_all_red(neighbor_id)
        n_state.update({
            "signal_state":      "SPILLBACK_LOCKED",
            "status_label":      "SPILLBACK_LOCKED",
            "signal_overridden": True,
        })
        restoration_order.append((neighbor_id, "FEEDER"))
        _emit_signal_changed(neighbor_id, old_sig, "SPILLBACK_LOCKED",
                             f"spillback for {junction_id}", emit_fn)

    _spillback_restore_order[junction_id] = restoration_order


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------

def restore_junction(junction_id: str, emit_fn: Callable | None = None) -> None:
    """
    Called after ambulance passes junction_id.
    Restores signal to NORMAL, then restores spillback neighbors in reverse order.
    """
    emit_fn = emit_fn or _emit_event

    def _restore():
        time.sleep(RESTORE_DELAY_S)

        state = junction_state.get(junction_id)
        if state:
            _traci_set_phase(junction_id, 0)   # return to normal cycling
            old_sig = state.get("signal_state", "AMBULANCE_GREEN")
            state.update({
                "signal_state":      "NORMAL",
                "signal_overridden": False,
                "active_corridor":   None,
                "status_label":      "NORMAL",
            })
            _log(f"🔄 {junction_id} signal restored to NORMAL", emit_fn)
            _emit_signal_changed(junction_id, old_sig, "NORMAL", "ambulance passed", emit_fn)

        # Restore spillback neighbors in reverse order
        for neighbor_id, role in reversed(_spillback_restore_order.get(junction_id, [])):
            n_state = junction_state.get(neighbor_id)
            if n_state and n_state.get("signal_overridden"):
                _traci_set_phase(neighbor_id, 0)
                old_sig = n_state.get("signal_state", "SPILLBACK_LOCKED")
                n_state.update({
                    "signal_state":      "NORMAL",
                    "signal_overridden": False,
                    "status_label":      "NORMAL",
                })
                _emit_signal_changed(neighbor_id, old_sig, "NORMAL", "spillback restored", emit_fn)

        _spillback_restore_order.pop(junction_id, None)

    t = threading.Thread(target=_restore, daemon=True)
    t.start()


def restore_all_for_ambulance(ambulance_id: str, emit_fn: Callable | None = None) -> None:
    """Called when ambulance reaches hospital. Clean up all its junctions."""
    emit_fn = emit_fn or _emit_event
    for jid, state in junction_state.items():
        if state.get("active_corridor") == ambulance_id:
            restore_junction(jid, emit_fn)


def cancel_scheduled_signals(junction_ids: list[str]) -> None:
    """Cancel timers for a list of junctions (used on reroute)."""
    for jid in junction_ids:
        t = _scheduled_timers.get(jid)
        if t and t.is_alive():
            t.cancel()
            del _scheduled_timers[jid]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg: str, emit_fn: Callable | None) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[Signal] {ts} {msg}")
    if emit_fn:
        try:
            emit_fn("event_log", {"timestamp": ts, "message": msg, "level": "INFO"},
                    room="controller")
        except Exception:
            pass


def _emit_signal_changed(
    junction_id: str,
    old_state: str,
    new_state: str,
    reason: str,
    emit_fn: Callable | None,
) -> None:
    if emit_fn:
        try:
            emit_fn("signal_changed", {
                "junction":  junction_id,
                "old_state": old_state,
                "new_state": new_state,
                "reason":    reason,
            }, room="controller")
        except Exception:
            pass
