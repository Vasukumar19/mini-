"""
setup.py — One-time setup script.
Run this ONCE before starting app.py for the first time:

    python setup.py

It will:
  1. Set up SUMO network files
  2. Train the LSTM model (50 epochs, ~30 days of synthetic traffic)
  3. Create junction_images directory
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def step(n, title):
    print("\n" + "=" * 50)
    print(f"  Step {n}: {title}")
    print("=" * 50)


def main():
    print("=" * 52)
    print("  Intelligent Ambulance System -- First-Time Setup")
    print("=" * 52)

    # -- Step 1: SUMO network --------------------------------------
    step(1, "Generating SUMO network files")
    cfg = os.path.join(THIS_DIR, "sumo_config", "city.sumocfg")
    if os.path.exists(cfg):
        print("  [OK] SUMO config already exists — skipping.")
    else:
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(THIS_DIR, "generate_sumo_network.py")],
            cwd=THIS_DIR,
        )
        if result.returncode != 0:
            print("  ✗ SUMO network generation failed. Check SUMO installation.")
            print("    You can still run app.py — it will use Python fallback mode.")
        else:
            print("  ✓ SUMO network ready.")

    # ── Step 2: LSTM training ─────────────────────────────────────
    step(2, "Training LSTM traffic model")
    model_path = os.path.join(THIS_DIR, "models", "lstm_traffic.pth")
    if os.path.exists(model_path):
        print("  ✓ LSTM weights already exist — skipping.")
        print(f"    ({model_path})")
    else:
        print("  Training on 30 days of synthetic traffic data …")
        sys.path.insert(0, THIS_DIR)
        from lstm_train import train
        train(epochs=50, n_days=30, patience=8)
        print("  ✓ LSTM model trained and saved.")

    # ── Step 3: junction_images directory ─────────────────────────
    step(3, "Preparing junction images directory")
    img_dir = os.path.join(THIS_DIR, "junction_images")
    os.makedirs(img_dir, exist_ok=True)

    # Check if original project has traffic images to copy
    old_traffic = os.path.abspath(
        os.path.join(THIS_DIR, "..", "mini project", "traffic_images")
    )
    if os.path.isdir(old_traffic):
        import shutil
        copied = 0
        for jnum in range(1, 18):
            j_dir = os.path.join(old_traffic, f"junction_J{jnum}")
            if not os.path.isdir(j_dir):
                continue
            for fname in os.listdir(j_dir):
                if fname.endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(j_dir, fname)
                    if os.path.getsize(src) == 0:
                        continue
                    # Rename to new scheme: J1_peak_morning.jpg
                    base, ext = os.path.splitext(fname)
                    new_name = f"J{jnum}_{base}{ext}"
                    dst = os.path.join(img_dir, new_name)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied += 1
        print(f"  ✓ Copied {copied} traffic images from previous project.")
    else:
        print("  ℹ  No existing traffic images found.")
        print("     YOLOv8 will use time-of-day synthetic density as fallback.")

    # ── Done ──────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Setup complete! Now run:                        ║")
    print("║                                                  ║")
    print("║    python app.py                                 ║")
    print("║                                                  ║")
    print("║  Then open:                                      ║")
    print("║    Controller: http://localhost:5000/controller  ║")
    print("║    Driver tab: http://localhost:5000/driver?id=ambu_01 ║")
    print("╚══════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
