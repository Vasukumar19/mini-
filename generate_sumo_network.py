"""
generate_sumo_network.py
One-time script to create SUMO simulation files.

How to run (requires SUMO to be installed):
    python generate_sumo_network.py

This generates:
    sumo_config/city.net.xml
    sumo_config/city.rou.xml
    sumo_config/city.sumocfg
"""

import os
import subprocess
import sys
import shutil

SUMO_CFG_DIR = "sumo_config"

# Try to locate existing SUMO install (from the original project)
CANDIDATE_SUMO_HOME = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mini project", "sumo_install", "sumo-1.19.0")
)

def setup_sumo_home():
    """Try to find and set SUMO_HOME from the existing project."""
    if os.path.exists(CANDIDATE_SUMO_HOME):
        os.environ["SUMO_HOME"] = CANDIDATE_SUMO_HOME
        sys.path.append(os.path.join(CANDIDATE_SUMO_HOME, "tools"))
        print(f"Using SUMO from: {CANDIDATE_SUMO_HOME}")
        return CANDIDATE_SUMO_HOME
    # Fallback: check system PATH
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        print(f"Using SUMO_HOME from environment: {sumo_home}")
        return sumo_home
    print("WARNING: SUMO_HOME not found. Trying system PATH.")
    return ""


def get_sumo_bin(binary: str, sumo_home: str) -> str:
    """Return full path to a SUMO binary, or just the name for PATH lookup."""
    if sumo_home:
        candidate = os.path.join(sumo_home, "bin", binary + ".exe")
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(sumo_home, "bin", binary)
        if os.path.exists(candidate):
            return candidate
    return binary


def copy_existing_network():
    """
    If the original project's network files exist, copy them instead of
    regenerating (avoids netgenerate dependency issues on Windows).
    Returns True if successful.
    """
    original_net = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mini project", "network")
    )
    if not os.path.isdir(original_net):
        return False

    os.makedirs(SUMO_CFG_DIR, exist_ok=True)

    files_to_copy = {
        "map.net.xml": "city.net.xml",
        "routes.rou.xml": "city.rou.xml",
    }

    copied = 0
    for src_name, dst_name in files_to_copy.items():
        src = os.path.join(original_net, src_name)
        dst = os.path.join(SUMO_CFG_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src_name} → {SUMO_CFG_DIR}/{dst_name}")
            copied += 1

    return copied > 0


def generate_with_netgenerate(sumo_home: str):
    """Generate a fresh grid network using SUMO netgenerate."""
    os.makedirs(SUMO_CFG_DIR, exist_ok=True)
    netgenerate = get_sumo_bin("netgenerate", sumo_home)
    net_out = os.path.join(SUMO_CFG_DIR, "city.net.xml")

    cmd = [
        netgenerate,
        "--grid",
        "--grid.x-number", "4",
        "--grid.y-number", "5",
        "--grid.x-length", "200",
        "--grid.y-length", "200",
        "--output-file", net_out,
        "--tls.guess", "true",
        "--no-warnings",
    ]
    print("Running netgenerate …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"netgenerate failed:\n{result.stderr}")
        return False
    print(f"Network generated → {net_out}")
    return True


def generate_routes(sumo_home: str):
    """Generate random background traffic routes."""
    net_xml = os.path.join(SUMO_CFG_DIR, "city.net.xml")
    rou_xml = os.path.join(SUMO_CFG_DIR, "city.rou.xml")

    # Try randomTrips.py from SUMO tools
    tools_dir = os.path.join(sumo_home, "tools") if sumo_home else ""
    random_trips = os.path.join(tools_dir, "randomTrips.py") if tools_dir else "randomTrips.py"

    if not os.path.exists(random_trips):
        # Write minimal manual routes file
        print("randomTrips.py not found. Writing minimal routes file.")
        write_minimal_routes(rou_xml)
        return

    cmd = [
        sys.executable, random_trips,
        "-n", net_xml,
        "-o", rou_xml,
        "--period", "3",
        "--end", "3600",
        "--validate",
    ]
    print("Generating routes …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"randomTrips failed, writing minimal routes: {result.stderr[:200]}")
        write_minimal_routes(rou_xml)
    else:
        print(f"Routes generated → {rou_xml}")


def write_minimal_routes(path: str):
    """Write a minimal routes XML with a few background vehicles."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>
    <vType id="ambulance" accel="3.5" decel="5.0" sigma="0.0" length="5.5"
           maxSpeed="22.22" color="1,1,1" guiShape="emergency"/>
</routes>
"""
    with open(path, "w") as f:
        f.write(xml)
    print(f"Minimal routes written → {path}")


def write_sumocfg():
    """Write the SUMO configuration file."""
    cfg_path = os.path.join(SUMO_CFG_DIR, "city.sumocfg")
    content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="city.net.xml"/>
    <route-files value="city.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="7200"/>
    <step-length value="0.5"/>
  </time>
  <report>
    <no-warnings value="true"/>
    <no-step-log value="true"/>
  </report>
</configuration>
"""
    with open(cfg_path, "w") as f:
        f.write(content)
    print(f"SUMO config written → {cfg_path}")


def main():
    sumo_home = setup_sumo_home()

    # Prefer copying existing working network files
    if copy_existing_network():
        print("Reused existing network files from mini project.")
    else:
        if not generate_with_netgenerate(sumo_home):
            print("ERROR: Could not generate network. "
                  "Copy map.net.xml manually to sumo_config/city.net.xml")
            return

    generate_routes(sumo_home)
    write_sumocfg()
    print("\n✅ SUMO setup complete. Files in:", SUMO_CFG_DIR)


if __name__ == "__main__":
    main()
