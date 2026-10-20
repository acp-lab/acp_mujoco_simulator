#!/usr/bin/env python3
"""Local MAVLink command client for PX4 SITL connected to MuJoCo.

The UDP command/telemetry connection is independent of the TCP simulator link.
All rate arguments and ROS inputs use body FLU, rad/s. MAVLink uses body FRD.
"""

import argparse
import csv
import math
import time
from pathlib import Path

from pymavlink import mavutil


class SitlClient:
    def __init__(self, port=14540):
        self.link = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{port}", source_system=255, source_component=190
        )
        self.heartbeat = None
        self.attitude = None
        self.position = None
        self.landed_state = None
        self.last_heartbeat_sent = -math.inf
        self.last_heartbeat_received = -math.inf
        self.acks = {}

    def poll(self):
        while True:
            msg = self.link.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_type() == "HEARTBEAT" and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4:
                self.heartbeat = msg
                self.link.target_system = msg.get_srcSystem()
                self.link.target_component = msg.get_srcComponent()
                self.last_heartbeat_received = time.monotonic()
            elif self.heartbeat and msg.get_srcSystem() == self.link.target_system:
                if msg.get_type() == "ATTITUDE":
                    self.attitude = msg
                elif msg.get_type() == "LOCAL_POSITION_NED":
                    self.position = msg
                elif msg.get_type() == "EXTENDED_SYS_STATE":
                    self.landed_state = msg.landed_state
                elif msg.get_type() == "COMMAND_ACK":
                    self.acks[msg.command] = msg.result
                elif msg.get_type() == "STATUSTEXT":
                    print(f"PX4: {msg.text}", flush=True)
        now = time.monotonic()
        if self.heartbeat and now - self.last_heartbeat_sent >= 1.0:
            self.link.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            self.last_heartbeat_sent = now

    def connect(self, timeout=20.0):
        deadline = time.monotonic() + timeout
        while self.heartbeat is None and time.monotonic() < deadline:
            self.poll()
            time.sleep(0.01)
        if self.heartbeat is None:
            raise RuntimeError("No PX4 heartbeat on localhost. Start SITL first.")
        print(f"Connected to PX4 system {self.link.target_system}", flush=True)
        for message_id in (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                           mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                           mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE):
            self.command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message_id, 20000)

    def command(self, command, *params):
        self.acks.pop(command, None)
        self.link.mav.command_long_send(
            self.link.target_system, self.link.target_component, command, 0,
            *(list(params) + [0.0] * (7 - len(params))),
        )

    def rates(self, thrust, roll=0.0, pitch=0.0, yaw=0.0):
        if not all(math.isfinite(v) for v in (thrust, roll, pitch, yaw)):
            raise ValueError("Rate/thrust commands must be finite")
        if not 0.0 <= thrust <= 1.0:
            raise ValueError("Normalized thrust must lie in [0, 1]")
        self.link.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.link.target_system, self.link.target_component,
            128,  # Ignore attitude quaternion; use all three body rates and thrust.
            [1.0, 0.0, 0.0, 0.0], roll, -pitch, -yaw, thrust,
        )

    def pump(self, seconds, thrust=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            if thrust is not None:
                self.rates(thrust)
            time.sleep(0.01)

    def enter_offboard_and_arm(self, thrust):
        # PX4 must receive setpoints before it accepts Offboard mode.
        self.pump(1.5, thrust)
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, 6, 0)
        self.pump(0.5, thrust)
        result = self.acks.get(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        if result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"Offboard mode not acknowledged: {result}")
        self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        self.pump(0.5, thrust)
        result = self.acks.get(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        if result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"Arming not acknowledged: {result}")

    def land_and_disarm(self):
        if self.heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            self.command(mavutil.mavlink.MAV_CMD_NAV_LAND)
            print("Requesting PX4 landing before disarming", flush=True)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                self.pump(0.1)
                if self.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND:
                    break
            else:
                raise RuntimeError("PX4 did not report landing within 45 seconds")
        self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
        self.pump(0.5)
        result = self.acks.get(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        if result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"Disarming not acknowledged: {result}")


def run_probe(client, args):
    if not args.arm:
        raise ValueError("The simulated flight probe requires --arm")
    if not 2.0 <= args.duration <= 30.0:
        raise ValueError("Use a probe duration between 2 and 30 seconds")
    fields = ["elapsed_s", "thrust", "roll_sp_flu", "pitch_sp_flu", "yaw_sp_flu",
              "roll_frd", "pitch_frd", "yaw_frd", "p_flu", "q_flu", "r_flu", "z_ned"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.enter_offboard_and_arm(args.thrust)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                elapsed = time.monotonic() - start
                client.poll()
                if time.monotonic() - client.last_heartbeat_received > 3:
                    raise RuntimeError("Lost PX4 heartbeat")
                # Give the vehicle time to leave the floor before the rate pulse.
                pulse = 0.2 if 1.5 <= elapsed < 2.5 else (-0.2 if 2.5 <= elapsed < 3.5 else 0.0)
                rates = [0.0, 0.0, 0.0]
                rates[{"roll": 0, "pitch": 1, "yaw": 2}[args.axis]] = pulse
                client.rates(args.thrust, *rates)
                a, p = client.attitude, client.position
                if a:
                    writer.writerow([elapsed, args.thrust, *rates, a.roll, a.pitch, a.yaw,
                                     a.rollspeed, -a.pitchspeed, -a.yawspeed,
                                     p.z if p else math.nan])
                time.sleep(0.01)
    finally:
        client.land_and_disarm()
    print(f"Probe finished; telemetry saved to {output}", flush=True)


def run_ros(client, args):
    import rclpy
    from quadrotor_msgs.msg import TRPYCommand

    if not math.isfinite(args.max_thrust) or args.max_thrust <= 0:
        raise ValueError("--max-thrust must be positive and match the plant")
    rclpy.init()
    node = rclpy.create_node("px4_mavlink_rates")
    latest = [None, -math.inf]

    def callback(msg):
        values = (msg.thrust, msg.angular_velocity.x,
                  msg.angular_velocity.y, msg.angular_velocity.z)
        if not all(math.isfinite(v) for v in values) or not 0 <= msg.thrust <= args.max_thrust:
            node.get_logger().error("Rejected invalid or out-of-range TRPY command")
            latest[0] = None
            return
        latest[:] = [values, time.monotonic()]

    subscription = node.create_subscription(TRPYCommand, args.topic, callback, 1)
    node.get_logger().info(
        f"{args.topic} -> PX4 body rates. Thrust scale: {args.max_thrust:.6f} N. "
        "Select Offboard mode and arm from the PX4 console after streaming commands."
    )
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            client.poll()
            if latest[0] is not None and time.monotonic() - latest[1] <= args.command_timeout:
                thrust, roll, pitch, yaw = latest[0]
                client.rates(thrust / args.max_thrust, roll, pitch, yaw)
            # Stop setpoints on stale input: allow PX4's Offboard-loss failsafe.
            time.sleep(0.005)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=14540)
    sub = parser.add_subparsers(dest="mode", required=True)
    monitor = sub.add_parser("monitor")
    monitor.add_argument("--duration", type=float, default=10)
    probe = sub.add_parser("probe", help="Short local SITL rate pulse, then PX4 landing")
    probe.add_argument("--arm", action="store_true")
    probe.add_argument("--thrust", type=float, required=True, help="Normalized collective [0,1]")
    probe.add_argument("--duration", type=float, default=5)
    probe.add_argument("--axis", choices=["roll", "pitch", "yaw"], default="yaw")
    probe.add_argument("--output", default="/tmp/px4_rate_probe.csv")
    ros = sub.add_parser("ros", help="Stream ROS 2 TRPY commands; never arms automatically")
    ros.add_argument("--topic", default="/quadrotor/trpy_cmd")
    ros.add_argument("--max-thrust", type=float, required=True, help="Collective at full command [N]")
    ros.add_argument("--command-timeout", type=float, default=0.2)
    args = parser.parse_args()
    client = SitlClient(args.port)
    try:
        client.connect()
        if args.mode == "probe":
            run_probe(client, args)
        elif args.mode == "ros":
            run_ros(client, args)
        else:
            for _ in range(max(1, int(args.duration))):
                client.pump(1)
                print(f"mode={mavutil.mode_string_v10(client.heartbeat)} "
                      f"attitude={client.attitude} position={client.position}", flush=True)
    finally:
        client.link.close()


if __name__ == "__main__":
    main()
