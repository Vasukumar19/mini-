"""
ambulance_state.py — AmbulanceState dataclass.
One instance per active ambulance. Stored in ambulance_registry.
"""

import time
from dataclasses import dataclass, field


@dataclass
class AmbulanceState:
    # Core identity
    id: str                     # e.g. "ambu_01"
    origin_junction: str        # e.g. "J5"
    severity: str               # "LOW" | "MEDIUM" | "HIGH"
    patient_count: int          # 1–10
    dispatch_time: float = field(default_factory=time.time)

    # Route planning (filled after A* runs)
    route: list = field(default_factory=list)           # [J5, J8, J9, J14, ...]
    hospital_id: str = ""
    hospital_name: str = ""
    estimated_travel_time: float = 0.0                  # seconds from A*

    # Runtime navigation state
    current_junction_index: int = 0
    status: str = "DISPATCHING"     # DISPATCHING | EN_ROUTE | HELD | ARRIVED

    # SUMO integration
    sumo_id: str = ""               # set post-init to f"ambulance_{id}"
    sumo_position: tuple = (0.0, 0.0)
    speed_kmh: float = 0.0
    distance_to_next: float = 0.0

    # Metrics
    signals_overridden: int = 0
    spillbacks_prevented: int = 0
    conflicts_resolved: int = 0
    reroute_count: int = 0
    time_saved: float = 0.0
    arrival_time: float = 0.0
    last_reroute: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.sumo_id:
            self.sumo_id = f"ambulance_{self.id}"

    # -------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------

    @property
    def current_junction(self) -> str:
        if self.route and self.current_junction_index < len(self.route):
            return self.route[self.current_junction_index]
        return self.origin_junction

    @property
    def next_junction(self) -> str | None:
        idx = self.current_junction_index + 1
        if self.route and idx < len(self.route):
            return self.route[idx]
        return None

    @property
    def waiting_time_seconds(self) -> float:
        return time.time() - self.dispatch_time

    @property
    def distance_to_hospital(self) -> int:
        """Remaining junctions on route."""
        return max(len(self.route) - self.current_junction_index, 1)

    def to_summary(self) -> dict:
        """Light dict for WebSocket payloads."""
        return {
            "id": self.id,
            "severity": self.severity,
            "patients": self.patient_count,
            "status": self.status,
            "current_junction": self.current_junction,
            "next_junction": self.next_junction,
            "hospital_name": self.hospital_name,
            "eta_seconds": self.estimated_travel_time,
            "signals_overridden": self.signals_overridden,
            "route": self.route,
        }
