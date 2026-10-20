#!/usr/bin/env python3
"""Run PX4 v1.16.0 SITL connected directly to the MuJoCo MAVLink simulator."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.instance <= 9:
        parser.error("--instance must lie in [0,9]")
    build = args.px4_dir.resolve() / "build/px4_sitl_default"
    binary = build / "bin/px4"
    if not binary.is_file():
        parser.error(f"Build PX4 first: missing {binary}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Avoid inheriting an unrelated vehicle's PX4 startup overrides.
    for name in list(env):
        if name.startswith("PX4_PARAM_") or name in (
                "PX4_SYS_AUTOSTART", "PX4_SIM_MODEL", "PX4_SIMULATOR",
                "PX4_SIM_HOSTNAME", "PX4_SIM_HOST_ADDR"):
            del env[name]
    env["PX4_SIM_MODEL"] = "none_iris"
    env["PX4_SIMULATOR"] = "mavlink"
    env["PX4_SIM_HOST_ADDR"] = "127.0.0.1"
    # The simulator link uses MAVLink; keep the unused DDS client off a default agent.
    env["PX4_UXRCE_DDS_PORT"] = "18888"
    env.setdefault("ROS_DOMAIN_ID", "147")
    config = Path(__file__).resolve().parent.parent / "config/px4_mujoco.json"
    for name, value in json.loads(config.read_text()).items():
        env[f"PX4_PARAM_{name}"] = str(value)
    print(f"Starting PX4 for MuJoCo: UDP offboard port {14540 + args.instance}; "
          f"TCP simulator port {4560 + args.instance}", flush=True)
    os.execve(str(binary), [
        str(binary), str(build / "etc"), "-w", str(args.run_dir.resolve()),
        "-i", str(args.instance),
    ], env)


if __name__ == "__main__":
    main()
