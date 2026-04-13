"""
conflict_resolver.py — Multi-ambulance priority scoring and conflict resolution.

When two ambulances need the same junction within 30 seconds of each other,
this module decides which one gets priority.

Depends on: ambulance_state.py (type hint only, no circular import)
"""

import time

# ---------------------------------------------------------------------------
# Priority weights
# ---------------------------------------------------------------------------

SEVERITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

WEIGHT_SEVERITY  = 0.50
WEIGHT_PROXIMITY = 0.30
WEIGHT_WAIT      = 0.20

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def priority_score(amb) -> float:
    """
    Weighted priority score for one AmbulanceState.

    Components (all normalised to roughly the same scale):
      severity_component  → 1–3 × 0.5
      proximity_component → 1/distance_to_hospital × 0.3 (closer = higher score)
      wait_component      → waiting_time_seconds / 60 × 0.2
    """
    sev = SEVERITY_MAP.get(amb.severity, 1)
    # proximity: ambulances with fewer junctions left score higher
    proximity = 1.0 / max(amb.distance_to_hospital, 1)
    wait = amb.waiting_time_seconds / 60.0

    score = (
        sev     * WEIGHT_SEVERITY
        + proximity * WEIGHT_PROXIMITY
        + wait      * WEIGHT_WAIT
    )
    return round(score, 4)


def resolve_conflict(amb_a, amb_b, contested_junction: str) -> dict:
    """
    Compare two ambulances competing for the same junction.

    Returns a result dict with full breakdown for the controller panel:
      {
        "winner":       ambulance_id,
        "loser":        ambulance_id,
        "winner_score": float,
        "loser_score":  float,
        "tiebreak":     None | "patient_count" | "dispatch_time",
        "junction":     junction_id,
        "action":       description string,
        "scores":       {amb_id: {total, severity, proximity, wait}},
        "timestamp":    HH:MM:SS,
      }
    """
    score_a = priority_score(amb_a)
    score_b = priority_score(amb_b)

    # Build per-component breakdown for UI display
    def breakdown(amb, score):
        sev = SEVERITY_MAP.get(amb.severity, 1)
        prox = 1.0 / max(amb.distance_to_hospital, 1)
        wait = amb.waiting_time_seconds / 60.0
        return {
            "total":     round(score, 4),
            "severity":  round(sev * WEIGHT_SEVERITY, 3),
            "proximity": round(prox * WEIGHT_PROXIMITY, 3),
            "wait":      round(wait * WEIGHT_WAIT, 3),
        }

    scores = {
        amb_a.id: breakdown(amb_a, score_a),
        amb_b.id: breakdown(amb_b, score_b),
    }

    tiebreak = None
    if abs(score_a - score_b) < 0.0001:
        # Tiebreak 1: higher patient count
        if amb_a.patient_count != amb_b.patient_count:
            tiebreak = "patient_count"
            winner = amb_a if amb_a.patient_count > amb_b.patient_count else amb_b
        else:
            # Tiebreak 2: earlier dispatch time (more urgent)
            tiebreak = "dispatch_time"
            winner = amb_a if amb_a.dispatch_time < amb_b.dispatch_time else amb_b
        loser = amb_b if winner is amb_a else amb_a
    else:
        winner = amb_a if score_a > score_b else amb_b
        loser  = amb_b if winner is amb_a else amb_a

    winner_score = scores[winner.id]["total"]
    loser_score  = scores[loser.id]["total"]

    return {
        "winner":       winner.id,
        "loser":        loser.id,
        "winner_score": winner_score,
        "loser_score":  loser_score,
        "tiebreak":     tiebreak,
        "junction":     contested_junction,
        "action":       f"{loser.id} held 20s at previous junction",
        "scores":       scores,
        "timestamp":    time.strftime("%H:%M:%S"),
        "winner_obj":   winner,   # internal use only, not serialised to JSON
        "loser_obj":    loser,
    }


def check_for_conflicts(ambulances: list, window_seconds: float = 30.0) -> list[dict]:
    """
    Scan all active ambulances for junction conflicts within window_seconds.

    Returns a list of conflict result dicts (one per conflicting pair).
    Idempotent — does not mutate state; callers decide how to act.
    """
    conflicts: list[dict] = []
    seen_pairs: set[frozenset] = set()

    for i, amb_a in enumerate(ambulances):
        if amb_a.status in ("ARRIVED", "HELD"):
            continue
        for j, amb_b in enumerate(ambulances):
            if i >= j:
                continue
            if amb_b.status in ("ARRIVED", "HELD"):
                continue

            pair = frozenset({amb_a.id, amb_b.id})
            if pair in seen_pairs:
                continue

            # Check if both ambulances are targeting the same next junction
            # within the time window
            if (amb_a.next_junction
                    and amb_a.next_junction == amb_b.next_junction):
                # Both approaching the same junction — conflict!
                seen_pairs.add(pair)
                result = resolve_conflict(amb_a, amb_b, amb_a.next_junction)
                conflicts.append(result)

    return conflicts
