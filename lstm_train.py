"""
lstm_train.py — Synthetic traffic data generation + LSTM training.
Run this script once: python lstm_train.py
Saves weights to models/lstm_traffic.pth
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Model definition (also imported by lstm_model.py)
# ---------------------------------------------------------------------------

class TrafficLSTM(nn.Module):
    """
    Input:  (batch, 10, 1) — last 10 density readings
    Output: (batch, 1)     — predicted next density, sigmoid-bounded [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq, 64)
        return self.sigmoid(self.fc(out[:, -1, :]))   # (batch, 1)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _hour_base_density(hour_frac: float) -> float:
    """Smooth sinusoidal traffic pattern with two peaks."""
    # Morning peak ~8h, evening peak ~18h
    morning = math.exp(-0.5 * ((hour_frac - 8) / 1.5) ** 2)
    evening = math.exp(-0.5 * ((hour_frac - 18) / 1.5) ** 2)
    night_dip = 0.05
    val = night_dip + 0.85 * (0.6 * morning + 0.9 * evening)
    return min(max(val, 0.0), 1.0)


def generate_synthetic_traffic(n_days: int = 60, tick_seconds: int = 10) -> list[float]:
    """
    Returns a flat list of density readings.
    n_days × 86400 / tick_seconds readings total.
    """
    ticks_per_day = 86400 // tick_seconds
    data: list[float] = []

    for day in range(n_days):
        # occasional incident day (2 per day on average)
        incident_ticks = sorted(random.sample(range(ticks_per_day),
                                              min(2, ticks_per_day)))
        active_incident: dict[int, int] = {}  # tick_start → duration
        for t in incident_ticks:
            active_incident[t] = random.randint(5, 20)

        incident_active = 0  # remaining ticks

        for tick in range(ticks_per_day):
            hour_frac = (tick * tick_seconds) / 3600.0
            base = _hour_base_density(hour_frac)

            # noise
            noisy = base + random.gauss(0, 0.05)

            # incident spike
            if tick in active_incident:
                incident_active = active_incident[tick]
            if incident_active > 0:
                noisy += 0.30
                incident_active -= 1

            data.append(min(max(noisy, 0.0), 1.0))

    return data


def make_sequences(data: list[float], seq_len: int = 10):
    """Sliding window: X=(seq_len readings), y=(next reading)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len])
    return (
        torch.tensor(X, dtype=torch.float32).unsqueeze(-1),  # (N, 10, 1)
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),  # (N, 1)
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 10,
    n_days: int = 60,
) -> None:
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "lstm_traffic.pth")

    print("Generating synthetic traffic data …")
    raw = generate_synthetic_traffic(n_days=n_days)
    X, y = make_sequences(raw)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training on {device} | {len(X_train)} train / {len(X_val)} val samples")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.5f})")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.5f}")
    print(f"Weights saved → {save_path}")


if __name__ == "__main__":
    train()
