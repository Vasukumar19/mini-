"""
ambulance_registry.py — Global registry of all active AmbulanceState objects.

Thread-safe. All modules read/write through this registry.
Depends on: ambulance_state.py
"""

import threading
from ambulance_state import AmbulanceState

_lock = threading.Lock()
_registry: dict[str, AmbulanceState] = {}


def register(amb: AmbulanceState) -> None:
    with _lock:
        _registry[amb.id] = amb


def get(ambulance_id: str) -> AmbulanceState | None:
    with _lock:
        return _registry.get(ambulance_id)


def get_all() -> list[AmbulanceState]:
    with _lock:
        return list(_registry.values())


def get_active() -> list[AmbulanceState]:
    """Return all ambulances that have not yet arrived."""
    with _lock:
        return [a for a in _registry.values() if a.status != "ARRIVED"]


def remove(ambulance_id: str) -> None:
    with _lock:
        _registry.pop(ambulance_id, None)


def count() -> int:
    with _lock:
        return len(_registry)


def summary_list() -> list[dict]:
    """Return light serialisable dicts for all ambulances."""
    with _lock:
        return [a.to_summary() for a in _registry.values()]
