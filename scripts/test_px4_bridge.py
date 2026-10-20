#!/usr/bin/env python3
"""Exercise the real MuJoCo/MAVLink bridge with a small fake PX4 peer.

Run after sourcing install/setup.bash, using the PX4 Python environment.
Tests wire data, physical motor torque signs, disarming, disconnect and timeout.
"""
import argparse
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from pymavlink.dialects.v20 import common as mavlink
import xacro


def exercise(plugins, scenario, rotor=0):
    share = Path(get_package_share_directory("acp_mujoco_simulator"))
    binary = Path(get_package_prefix("acp_mujoco_simulator")) / "lib/acp_mujoco_simulator/mujoco_px4"
    with tempfile.TemporaryDirectory(prefix="acp_px4_wire_test_") as temp:
        root = Path(temp)
        doc = xacro.process_file(
            str(share / "model/scene_px4.xml.xacro"),
            mappings={"init_z": "0.03" if scenario == "rest" else "2.0"},
        )
        model = root / "scene.xml"
        model.write_text(doc.toxml())
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        env = dict(os.environ, ROS_DOMAIN_ID="148", ROS_LOG_DIR=str(root / "ros_logs"))
        duration = 0.6 if scenario == "rest" else 0.16
        with (root / "runner.log").open("w+") as log:
            process = subprocess.Popen([
                str(binary), "--ros-args", "-p", f"model_path:={model}",
                "-p", f"mujoco_plugin_dir:={plugins}", "-p", f"simulator_port:={port}",
                "-p", f"duration_s:={duration}",
            ], env=env, stdout=log, stderr=subprocess.STDOUT)
            client = socket.socket()
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        client.connect(("127.0.0.1", port))
                        break
                    except ConnectionRefusedError:
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("Bridge failed to start")
                        time.sleep(0.02)
                client.settimeout(5)
                decoder = mavlink.MAVLink(None)
                encoder = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
                samples = []
                stopped = False
                while True:
                    data = client.recv(4096)
                    if not data:
                        break
                    for message in decoder.parse_buffer(data) or []:
                        if message.get_type() != "HIL_SENSOR":
                            continue
                        samples.append(message)
                        u = [0.0] * 16
                        armed = 128
                        if scenario == "rotor":
                            u[rotor] = 0.25
                        elif scenario == "disarmed":
                            u[:4] = [0.7] * 4
                            armed = 0
                        elif scenario == "nonfinite":
                            u[0] = math.nan
                        if scenario == "disconnect":
                            client.close()
                            stopped = True
                            break
                        if scenario == "timeout" and len(samples) > 1:
                            continue
                        packet = mavlink.MAVLink_hil_actuator_controls_message(
                            message.time_usec, u, armed, 1
                        ).pack(encoder)
                        # Split a frame to verify TCP parsing across reads.
                        client.sendall(packet[:7])
                        client.sendall(packet[7:])
                    if stopped:
                        break
                code = process.wait(timeout=5)
                log.seek(0)
                output = log.read()
                if scenario in ("timeout", "disconnect", "nonfinite"):
                    assert code == 1, (scenario, code, output)
                    expected = {"timeout": "actuator timeout", "disconnect": "disconnected",
                                "nonfinite": "nonfinite"}[scenario]
                    assert expected in output, output
                else:
                    assert code == 0, output
                    assert len(samples) > 20
                    times = [s.time_usec for s in samples]
                    assert all(b > a for a, b in zip(times, times[1:]))
                    assert all(b - a == 4000 for a, b in zip(times[1:], times[2:]))
                    last = samples[-1]
                    if scenario == "rest":
                        assert abs(last.zacc + 9.81) < 0.1, last
                        assert abs(last.xacc) < 0.1 and abs(last.yacc) < 0.1
                    elif scenario == "disarmed":
                        # Free fall has only a small aerodynamic drag force,
                        # even if a disarmed peer sends nonzero motor commands.
                        assert abs(last.zacc) < 0.3, last
                    elif scenario == "rotor":
                        # Expected signs in FRD from the actual rotor sites/gears.
                        signs = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)][rotor]
                        rates = [last.xgyro, last.ygyro, last.zgyro]
                        assert all(sign * rate > 0.01 for sign, rate in zip(signs, rates)), (rotor, rates)
                print(f"PASS {scenario}" + (f" motor {rotor + 1}" if scenario == "rotor" else ""))
            finally:
                client.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugins", required=True)
    args = parser.parse_args()
    exercise(args.plugins, "rest")
    exercise(args.plugins, "disarmed")
    for rotor in range(4):
        exercise(args.plugins, "rotor", rotor)
    for scenario in ("nonfinite", "disconnect", "timeout"):
        exercise(args.plugins, scenario)


if __name__ == "__main__":
    main()
