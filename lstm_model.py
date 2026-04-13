"""
lstm_model.py — Load trained LSTM and expose predict_future_density().
Depends on: lstm_train.py (for TrafficLSTM class), junction_graph.py
"""

import os
import torch
from lstm_train import TrafficLSTM
from junction_graph import junction_state

# ---------------------------------------------------------------------------
# Load model weights once at import time
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lstm_traffic.pth")

_model = TrafficLSTM()
_model_loaded = False

def _load_model():
    global _model, _model_loaded
    if os.path.exists(_MODEL_PATH):
        try:
            _model.load_state_dict(
                torch.load(_MODEL_PATH, weights_only=True, map_location="cpu")
            )
            _model.eval()
            _model_loaded = True
            print(f"[LSTM] Weights loaded from {_MODEL_PATH}")
        except Exception as e:
            print(f"[LSTM] WARNING: Failed to load weights ({e}). Using untrained model as fallback.")
            _model.eval()
    else:
        print(f"[LSTM] WARNING: {_MODEL_PATH} not found. "
              "Run 'python lstm_train.py' first. Using fallback predictions.")

_load_model()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_future_density(junction_id: str, seconds_ahead: float) -> float:
    """
    Iteratively predict traffic density at a future timestamp.

    Uses the last 10 CV readings stored in junction_state[junction_id]["density_history"]
    as the initial input window, then rolls the LSTM forward step-by-step
    (each step = 10 seconds) until reaching seconds_ahead.

    Fallback: if model is not loaded, returns current_density * 1.1 (slight increase).

    Args:
        junction_id: e.g. "J5"
        seconds_ahead: how far in the future to predict (e.g. 90.0)

    Returns:
        Predicted density in [0.0, 1.0]
    """
    state = junction_state.get(junction_id)
    if state is None:
        return 0.5

    # Fallback if model weights are unavailable
    if not _model_loaded:
        return min(state["current_density"] * 1.1, 1.0)

    # Build sliding window from stored history
    history: list[float] = list(state["density_history"][-10:])
    while len(history) < 10:
        history.insert(0, 0.3)           # pad with neutral density

    steps_ahead = max(1, int(round(seconds_ahead / 10.0)))

    with torch.no_grad():
        window = list(history)
        for _ in range(steps_ahead):
            x = torch.tensor(window[-10:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            # x shape: (1, 10, 1)
            predicted = _model(x).item()       # single float in [0, 1]
            window.append(predicted)

    result = float(window[-1])
    return min(max(result, 0.0), 1.0)


def get_current_density(junction_id: str) -> float:
    """Return the latest measured density (no prediction)."""
    state = junction_state.get(junction_id)
    if state is None:
        return 0.3
    return state.get("current_density", 0.3)
