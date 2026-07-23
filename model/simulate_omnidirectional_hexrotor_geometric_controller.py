#!/usr/bin/env python3
"""Track a desired 6-DOF pose with the tilted hexrotor MuJoCo model.

This script generates the same visual RotorMotor-based omnidirectional
hexrotor model used by simulate_omnidirectional_hexrotor_wrench_mapping.py, but
enables gravity and closes a geometric pose controller around MuJoCo. The
controller computes a desired body wrench

    wrench_B = [F_B, M_B]

then allocates it to nonnegative motor forces and converts those forces to
rotor-speed commands using

    f_i = kf * omega_i^2.

All state, reference, wrench, allocation, and motor data are saved to disk.
"""

from __future__ import annotations

import argparse
import csv
import os
import time as wall_time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from compare_quadrotor_actuator_wrenches import actuator_matrix, load_actuator_wrenches
from simulate_omnidirectional_hexrotor_wrench_mapping import (
    DEFAULT_KF,
    DEFAULT_PLUGIN,
    THIS_DIR,
    WRENCH_NAMES,
    activation_by_actuator,
    delayed_command,
    generated_xml,
    load_plugin,
    make_tilted_hex_geometry,
    make_bounded_motor_force_allocator,
    preload_allocator_backend,
    save_matrix,
    set_activation_by_actuator,
)


DEFAULT_OUTPUT_DIR = THIS_DIR / "omnidirectional_hexrotor_geometric_tracking_results"


@dataclass(frozen=True)
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    rotation: np.ndarray
    omega_body: np.ndarray
    omega_dot_body: np.ndarray
    rpy: np.ndarray


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in np.asarray(values, dtype=float)) + "]"


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def vee(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            matrix[2, 1],
            matrix[0, 2],
            matrix[1, 0],
        ],
        dtype=float,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=float,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                ],
                dtype=float,
            )
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                ],
                dtype=float,
            )
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=float,
            )
    return quat / np.linalg.norm(quat)


def desired_position_terms(time: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(args.reference_position_center, dtype=float)
    amplitude = np.asarray(args.reference_position_amplitude, dtype=float)
    frequency = np.asarray(args.reference_position_frequency, dtype=float)
    phase = np.asarray(args.reference_position_phase, dtype=float)
    omega = 2.0 * np.pi * frequency
    angle = omega * time + phase
    position = center + amplitude * np.sin(angle)
    velocity = amplitude * omega * np.cos(angle)
    acceleration = -amplitude * omega * omega * np.sin(angle)
    return position, velocity, acceleration


def desired_orientation_matrix(time: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.asarray(args.reference_attitude_amplitude, dtype=float)
    frequency = np.asarray(args.reference_attitude_frequency, dtype=float)
    phase = np.asarray(args.reference_attitude_phase, dtype=float)
    rpy = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    return rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])), rpy


def desired_omega_body(time: float, args: argparse.Namespace) -> np.ndarray:
    step = args.orientation_diff_step
    rotation_plus, _ = desired_orientation_matrix(time + step, args)
    rotation_minus, _ = desired_orientation_matrix(time - step, args)
    rotation, _ = desired_orientation_matrix(time, args)
    rotation_dot = (rotation_plus - rotation_minus) / (2.0 * step)
    return vee(rotation.T @ rotation_dot)


def reference_at(time: float, args: argparse.Namespace) -> Reference:
    position, velocity, acceleration = desired_position_terms(time, args)
    rotation, rpy = desired_orientation_matrix(time, args)
    omega_body = desired_omega_body(time, args)
    step = args.orientation_diff_step
    omega_dot_body = (desired_omega_body(time + step, args) - desired_omega_body(time - step, args)) / (2.0 * step)
    return Reference(position, velocity, acceleration, rotation, omega_body, omega_dot_body, rpy)


def geometric_pose_controller(
    position: np.ndarray,
    velocity_world: np.ndarray,
    rotation: np.ndarray,
    omega_body: np.ndarray,
    reference: Reference,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kp_position = np.asarray(args.kp_position, dtype=float)
    kd_position = np.asarray(args.kd_position, dtype=float)
    kr_attitude = np.asarray(args.kr_attitude, dtype=float)
    kw_attitude = np.asarray(args.kw_attitude, dtype=float)
    inertia = np.diag(np.asarray(args.inertia, dtype=float))

    position_error = position - reference.position
    velocity_error = velocity_world - reference.velocity
    gravity_world = np.array([0.0, 0.0, args.gravity_z], dtype=float)
    desired_acceleration = reference.acceleration - kp_position * position_error - kd_position * velocity_error
    desired_force_world = args.mass * (desired_acceleration - gravity_world)
    desired_force_body = rotation.T @ desired_force_world

    attitude_error_matrix = 0.5 * (reference.rotation.T @ rotation - rotation.T @ reference.rotation)
    attitude_error = vee(attitude_error_matrix)
    desired_omega_in_current_body = rotation.T @ reference.rotation @ reference.omega_body
    omega_error = omega_body - desired_omega_in_current_body
    feedforward = (
        hat(omega_body) @ rotation.T @ reference.rotation @ reference.omega_body
        - rotation.T @ reference.rotation @ reference.omega_dot_body
    )
    desired_moment_body = (
        -kr_attitude * attitude_error
        - kw_attitude * omega_error
        + np.cross(omega_body, inertia @ omega_body)
        - inertia @ feedforward
    )
    wrench_body = np.concatenate((desired_force_body, desired_moment_body))
    return wrench_body, position_error, attitude_error, omega_error


def allocation_weights(args: argparse.Namespace) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(args.allocation_force_weights, dtype=float),
            np.asarray(args.allocation_moment_weights, dtype=float),
        )
    )


def set_initial_state(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace) -> None:
    reference = reference_at(0.0, args)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "omni_1")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "omni_1")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    joint_type = int(model.jnt_type[joint_id])
    position_offset = np.asarray(args.initial_position_offset, dtype=float)
    rpy_offset = np.asarray(args.initial_rpy_offset, dtype=float)
    initial_rotation = reference.rotation @ rpy_to_matrix(
        float(rpy_offset[0]), float(rpy_offset[1]), float(rpy_offset[2])
    )
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        data.qpos[qpos_adr : qpos_adr + 3] = reference.position + position_offset
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = matrix_to_quat(initial_rotation)
        data.qvel[qvel_adr : qvel_adr + 3] = reference.velocity
        data.qvel[qvel_adr + 3 : qvel_adr + 6] = reference.omega_body
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        if np.linalg.norm(position_offset) > 1.0e-12:
            raise ValueError("--initial-position-offset requires --joint-type free.")
        data.qpos[qpos_adr : qpos_adr + 4] = matrix_to_quat(initial_rotation)
        data.qvel[qvel_adr : qvel_adr + 3] = reference.omega_body
    else:
        raise ValueError("The controller supports free or ball joints only.")
    mujoco.mj_forward(model, data)
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) and body_id >= 0:
        args.reference_position_center = tuple(np.asarray(data.xpos[body_id], dtype=float))


def body_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "omni_1")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "omni_1")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        qpos = data.qpos[qpos_adr : qpos_adr + 7].copy()
        qvel = data.qvel[qvel_adr : qvel_adr + 6].copy()
        position = qpos[:3]
        quat = qpos[3:7]
        velocity_world = qvel[:3]
        omega_body = qvel[3:6]
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        position = data.xpos[body_id].copy()
        quat = data.qpos[qpos_adr : qpos_adr + 4].copy()
        velocity_world = np.zeros(3)
        omega_body = data.qvel[qvel_adr : qvel_adr + 3].copy()
    else:
        raise ValueError("The controller supports free or ball joints only.")
    rotation = quat_to_matrix(quat)
    return position, quat, velocity_world, omega_body, rotation


def initialize_motor_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    allocator: object,
    args: argparse.Namespace,
) -> np.ndarray:
    position, _, velocity, omega_body, rotation = body_state(model, data)
    reference = reference_at(0.0, args)
    wrench_body, _, _, _ = geometric_pose_controller(position, velocity, rotation, omega_body, reference, args)
    motor_force, _, _ = allocator.solve(wrench_body)
    omega_actual = np.sqrt(np.maximum(motor_force, 0.0) / args.kf)
    if args.actuator_model == "plugin":
        set_activation_by_actuator(model, data, omega_actual)
        data.ctrl[:] = omega_actual
    else:
        data.ctrl[:] = motor_force
    mujoco.mj_forward(model, data)
    return omega_actual


def launch_passive_viewer(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData):
    if not args.viewer:
        return None
    try:
        import mujoco.viewer as mujoco_viewer
    except Exception as exc:
        raise RuntimeError(
            "Could not import mujoco.viewer. Install the MuJoCo Python viewer dependencies "
            "and run from a desktop/OpenGL session."
        ) from exc

    try:
        viewer = mujoco_viewer.launch_passive(model, data)
    except Exception as exc:
        raise RuntimeError(
            "Could not launch the MuJoCo Python viewer. This usually means DISPLAY/OpenGL "
            "is not available in the current terminal session."
        ) from exc

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "omni_1")
    if body_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = args.viewer_distance
        viewer.cam.azimuth = args.viewer_azimuth
        viewer.cam.elevation = args.viewer_elevation
    elif camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
    viewer.sync()
    return viewer


def sync_viewer(
    viewer,
    args: argparse.Namespace,
    model: mujoco.MjModel,
    step: int,
    nsteps: int,
    step_start_wall_time: float,
) -> bool:
    if viewer is None:
        return True
    if not viewer.is_running():
        return False
    sync_interval = max(1, int(args.viewer_sync_interval))
    if step % sync_interval == 0 or step == nsteps:
        viewer.sync()
    if args.viewer_realtime:
        speed = max(float(args.viewer_speed), 1.0e-6)
        target_period = float(model.opt.timestep) / speed
        sleep_time = target_period - (wall_time.monotonic() - step_start_wall_time)
        if sleep_time > 0.0:
            wall_time.sleep(sleep_time)
    return True


def run_tracking(args: argparse.Namespace) -> dict[str, np.ndarray | str | float]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preload_allocator_backend(args.allocator)
    if args.actuator_model == "plugin":
        if args.tau <= 0.0:
            raise ValueError("RotorMotor plugin mode requires --tau > 0.")
        load_plugin(args.plugin_library)

    geometry = make_tilted_hex_geometry(args.radius, args.tilt_deg, args.axis_offset_deg, args.km_over_kf)
    xml_text = generated_xml(args, geometry)
    model_path = output_dir / "omnidirectional_hexrotor_geometric_tracking.xml"
    model_path.write_text(xml_text, encoding="utf-8")

    model, wrenches = load_actuator_wrenches(model_path, "omni_1")
    model.opt.gravity[:] = (0.0, 0.0, args.gravity_z)
    data = mujoco.MjData(model)
    set_initial_state(model, data, args)

    allocation = actuator_matrix(wrenches)
    omega_squared_allocation = allocation @ np.diag(np.full(model.nu, args.kf, dtype=float))
    force_max_scalar = args.kf * args.omega_max * args.omega_max
    force_min = np.zeros(model.nu)
    force_max = np.full(model.nu, force_max_scalar)
    allocator = make_bounded_motor_force_allocator(
        allocation,
        force_min,
        force_max,
        weights=allocation_weights(args),
        allocator=args.allocator,
        regularization=args.allocation_regularization,
        osqp_eps_abs=args.osqp_eps_abs,
        osqp_eps_rel=args.osqp_eps_rel,
        osqp_max_iter=args.osqp_max_iter,
        osqp_polish=args.osqp_polish,
        osqp_verbose=args.osqp_verbose,
    )
    omega_actual = initialize_motor_state(model, data, allocator, args)
    command_history: list[tuple[float, np.ndarray]] = [(0.0, omega_actual.copy())]
    viewer = launch_passive_viewer(args, model, data)

    nsteps = int(np.ceil(args.duration / model.opt.timestep))
    csv_path = output_dir / "omnidirectional_hexrotor_geometric_tracking.csv"
    allocator_status_counts: dict[str, int] = {}

    logs: dict[str, list[np.ndarray | float]] = {
        "time": [],
        "position": [],
        "position_desired": [],
        "velocity": [],
        "velocity_desired": [],
        "quaternion": [],
        "quaternion_desired": [],
        "rpy": [],
        "rpy_desired": [],
        "omega_body": [],
        "omega_body_desired": [],
        "position_error": [],
        "attitude_error": [],
        "omega_error": [],
        "desired_wrench": [],
        "command_wrench": [],
        "applied_wrench": [],
        "allocation_residual": [],
        "command_motor_force": [],
        "command_motor_omega": [],
        "actual_motor_omega": [],
        "applied_motor_force": [],
    }

    header = ["step", "time"]
    header += [f"position_{axis}" for axis in ("x", "y", "z")]
    header += [f"position_desired_{axis}" for axis in ("x", "y", "z")]
    header += [f"velocity_{axis}" for axis in ("x", "y", "z")]
    header += [f"velocity_desired_{axis}" for axis in ("x", "y", "z")]
    header += [f"quaternion_{axis}" for axis in ("w", "x", "y", "z")]
    header += [f"quaternion_desired_{axis}" for axis in ("w", "x", "y", "z")]
    header += [f"rpy_{axis}" for axis in ("roll", "pitch", "yaw")]
    header += [f"rpy_desired_{axis}" for axis in ("roll", "pitch", "yaw")]
    header += [f"omega_body_{axis}" for axis in ("x", "y", "z")]
    header += [f"omega_body_desired_{axis}" for axis in ("x", "y", "z")]
    header += [f"position_error_{axis}" for axis in ("x", "y", "z")]
    header += [f"attitude_error_{axis}" for axis in ("x", "y", "z")]
    header += [f"omega_error_{axis}" for axis in ("x", "y", "z")]
    header += [f"desired_wrench_{name}" for name in WRENCH_NAMES]
    header += [f"command_wrench_{name}" for name in WRENCH_NAMES]
    header += [f"applied_wrench_{name}" for name in WRENCH_NAMES]
    header += [f"allocation_residual_{name}" for name in WRENCH_NAMES]
    header += [f"motor_force_cmd_{i + 1}" for i in range(model.nu)]
    header += [f"omega_cmd_{i + 1}" for i in range(model.nu)]
    header += [f"omega_actual_{i + 1}" for i in range(model.nu)]
    header += [f"motor_force_applied_{i + 1}" for i in range(model.nu)]

    try:
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)

            for step in range(nsteps + 1):
                step_start_wall_time = wall_time.monotonic()
                time = float(data.time)
                position, quat, velocity_world, omega_body, rotation = body_state(model, data)
                reference = reference_at(time, args)
                desired_wrench, position_error, attitude_error, omega_error = geometric_pose_controller(
                    position, velocity_world, rotation, omega_body, reference, args
                )
                motor_force, allocation_residual, allocator_status = allocator.solve(desired_wrench)
                allocator_status_counts[allocator_status] = allocator_status_counts.get(allocator_status, 0) + 1
                omega_cmd = np.sqrt(np.maximum(motor_force, 0.0) / args.kf)

                if args.actuator_model == "plugin":
                    data.ctrl[:] = omega_cmd
                else:
                    command_history.append((time, omega_cmd.copy()))
                    delayed_omega_cmd = delayed_command(time, args.delay, command_history, command_history[0][1])
                    if args.tau <= 0.0:
                        omega_actual = delayed_omega_cmd.copy()
                    else:
                        alpha = np.exp(-float(model.opt.timestep) / args.tau)
                        omega_actual = alpha * omega_actual + (1.0 - alpha) * delayed_omega_cmd
                    data.ctrl[:] = args.kf * omega_actual * omega_actual

                mujoco.mj_forward(model, data)
                actuator_force = data.actuator_force.copy()
                if args.actuator_model == "plugin":
                    omega_actual = activation_by_actuator(model, data)
                command_wrench = allocation @ motor_force
                applied_wrench = allocation @ actuator_force
                quat_desired = matrix_to_quat(reference.rotation)
                rpy = matrix_to_rpy(rotation)

                row = (
                    [step, time]
                    + position.tolist()
                    + reference.position.tolist()
                    + velocity_world.tolist()
                    + reference.velocity.tolist()
                    + quat.tolist()
                    + quat_desired.tolist()
                    + rpy.tolist()
                    + reference.rpy.tolist()
                    + omega_body.tolist()
                    + reference.omega_body.tolist()
                    + position_error.tolist()
                    + attitude_error.tolist()
                    + omega_error.tolist()
                    + desired_wrench.tolist()
                    + command_wrench.tolist()
                    + applied_wrench.tolist()
                    + allocation_residual.tolist()
                    + motor_force.tolist()
                    + omega_cmd.tolist()
                    + omega_actual.tolist()
                    + actuator_force.tolist()
                )
                writer.writerow(row)

                logs["time"].append(time)
                logs["position"].append(position)
                logs["position_desired"].append(reference.position)
                logs["velocity"].append(velocity_world)
                logs["velocity_desired"].append(reference.velocity)
                logs["quaternion"].append(quat)
                logs["quaternion_desired"].append(quat_desired)
                logs["rpy"].append(rpy)
                logs["rpy_desired"].append(reference.rpy)
                logs["omega_body"].append(omega_body)
                logs["omega_body_desired"].append(reference.omega_body)
                logs["position_error"].append(position_error)
                logs["attitude_error"].append(attitude_error)
                logs["omega_error"].append(omega_error)
                logs["desired_wrench"].append(desired_wrench)
                logs["command_wrench"].append(command_wrench)
                logs["applied_wrench"].append(applied_wrench)
                logs["allocation_residual"].append(allocation_residual)
                logs["command_motor_force"].append(motor_force)
                logs["command_motor_omega"].append(omega_cmd)
                logs["actual_motor_omega"].append(omega_actual.copy())
                logs["applied_motor_force"].append(actuator_force)

                if step < nsteps:
                    mujoco.mj_step(model, data)
                if not sync_viewer(viewer, args, model, step, nsteps, step_start_wall_time):
                    break

        if viewer is not None and args.viewer_hold:
            while viewer.is_running():
                viewer.sync()
                wall_time.sleep(0.05)
    finally:
        if viewer is not None:
            viewer.close()

    if not logs["time"]:
        raise RuntimeError("No simulation samples were recorded.")

    result: dict[str, np.ndarray | str | float] = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "allocation": allocation,
        "omega_squared_allocation": omega_squared_allocation,
        "singular_values": np.linalg.svd(allocation, compute_uv=False),
        "condition_number": float(np.linalg.cond(allocation)),
        "rank": int(np.linalg.matrix_rank(allocation)),
        "force_max": force_max_scalar,
        "omega_max": args.omega_max,
        "kf": args.kf,
        "tau": args.tau,
        "delay": args.delay,
        "gravity_z": args.gravity_z,
        "joint_type": args.joint_type,
        "allocator": allocator.name,
        "allocation_regularization": args.allocation_regularization,
        "allocator_status_counts": allocator_status_counts,
        "allocation_weights": allocation_weights(args),
        "actuator_model": args.actuator_model,
        "plugin_library": str(args.plugin_library),
    }
    for key, value in logs.items():
        result[key] = np.asarray(value)
    return result


def write_summary(output_dir: Path, result: dict[str, np.ndarray | str | float], args: argparse.Namespace) -> None:
    time = np.asarray(result["time"])
    position_error = np.asarray(result["position_error"])
    attitude_error = np.asarray(result["attitude_error"])
    omega_error = np.asarray(result["omega_error"])
    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    residual = np.asarray(result["allocation_residual"])
    moment_error = residual[:, 3:6]
    command_force = np.asarray(result["command_motor_force"])
    command_omega = np.asarray(result["command_motor_omega"])
    actual_omega = np.asarray(result["actual_motor_omega"])
    applied_force = np.asarray(result["applied_motor_force"])
    velocity = np.asarray(result["velocity"])
    desired_velocity = np.asarray(result["velocity_desired"])
    speed = np.linalg.norm(velocity, axis=1)
    desired_speed = np.linalg.norm(desired_velocity, axis=1)
    metrics_mask = time >= args.metrics_start_time
    if not np.any(metrics_mask):
        metrics_mask = np.ones_like(time, dtype=bool)

    lines = [
        "Omnidirectional hexrotor geometric pose tracking",
        "",
        f"model xml: {result['model_path']}",
        f"csv: {result['csv_path']}",
        f"actuator model: {result['actuator_model']}",
        f"plugin library: {result['plugin_library']}",
        f"joint type: {result['joint_type']}",
        f"gravity z: {float(result['gravity_z']):.6g} m/s^2",
        f"mass: {args.mass:.6g} kg",
        f"inertia diagonal: {_fmt(np.asarray(args.inertia))} kg m^2",
        f"kf: {float(result['kf']):.16g} N/(rad/s)^2",
        f"omega max: {float(result['omega_max']):.6g} rad/s",
        f"motor force max: {float(result['force_max']):.6g} N",
        f"motor speed tau: {float(result['tau']):.6g} s",
        f"motor speed delay: {float(result['delay']):.6g} s",
        f"allocator: {result['allocator']}",
        f"allocation regularization: {float(result['allocation_regularization']):.6g}",
        f"allocator status counts: {result['allocator_status_counts']}",
        "",
        "Controller:",
        f"kp_position: {_fmt(np.asarray(args.kp_position))}",
        f"kd_position: {_fmt(np.asarray(args.kd_position))}",
        f"kr_attitude: {_fmt(np.asarray(args.kr_attitude))}",
        f"kw_attitude: {_fmt(np.asarray(args.kw_attitude))}",
        f"allocation weights [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.asarray(result['allocation_weights']))}",
        "",
        f"allocation rank: {int(result['rank'])}",
        f"allocation condition number: {float(result['condition_number']):.6g}",
        f"allocation singular values: {_fmt(np.asarray(result['singular_values']))}",
        "",
        "Allocation matrix A_force, rows [Fx, Fy, Fz, Mx, My, Mz], columns motor forces:",
        np.array2string(np.asarray(result["allocation"]), precision=6, suppress_small=True),
        "",
        "Direct omega^2 matrix A_omega2 = A_force @ diag(kf), rows [Fx, Fy, Fz, Mx, My, Mz]:",
        np.array2string(np.asarray(result["omega_squared_allocation"]), precision=12, suppress_small=False),
        "",
        f"max desired speed norm [m/s]: {float(np.max(desired_speed)):.6g}",
        f"max actual speed norm [m/s]: {float(np.max(speed)):.6g}",
        f"95th percentile actual speed norm [m/s]: {float(np.percentile(speed, 95.0)):.6g}",
        "",
        f"rms position error [m]: {float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))):.6g}",
        f"max position error [m]: {float(np.max(np.linalg.norm(position_error, axis=1))):.6g}",
        f"rms attitude error norm: {float(np.sqrt(np.mean(np.sum(attitude_error**2, axis=1)))):.6g}",
        f"max attitude error norm: {float(np.max(np.linalg.norm(attitude_error, axis=1))):.6g}",
        f"rms attitude error norm after {args.metrics_start_time:.6g} s: {float(np.sqrt(np.mean(np.sum(attitude_error[metrics_mask]**2, axis=1)))):.6g}",
        f"max attitude error norm after {args.metrics_start_time:.6g} s: {float(np.max(np.linalg.norm(attitude_error[metrics_mask], axis=1))):.6g}",
        f"rms omega error [rad/s]: {float(np.sqrt(np.mean(np.sum(omega_error**2, axis=1)))):.6g}",
        f"max omega error [rad/s]: {float(np.max(np.linalg.norm(omega_error, axis=1))):.6g}",
        f"rms omega error after {args.metrics_start_time:.6g} s [rad/s]: {float(np.sqrt(np.mean(np.sum(omega_error[metrics_mask]**2, axis=1)))):.6g}",
        f"max omega error after {args.metrics_start_time:.6g} s [rad/s]: {float(np.max(np.linalg.norm(omega_error[metrics_mask], axis=1))):.6g}",
        "",
        f"motor force command min: {_fmt(np.min(command_force, axis=0))}",
        f"motor force command max: {_fmt(np.max(command_force, axis=0))}",
        f"omega command min: {_fmt(np.min(command_omega, axis=0))}",
        f"omega command max: {_fmt(np.max(command_omega, axis=0))}",
        f"actual omega min: {_fmt(np.min(actual_omega, axis=0))}",
        f"actual omega max: {_fmt(np.max(actual_omega, axis=0))}",
        f"applied motor force min: {_fmt(np.min(applied_force, axis=0))}",
        f"applied motor force max: {_fmt(np.max(applied_force, axis=0))}",
        "",
        f"max abs allocation residual [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(residual), axis=0))}",
        f"max abs moment allocation residual [Mx, My, Mz]: {_fmt(np.max(np.abs(moment_error), axis=0))}",
        f"rms command wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.sqrt(np.mean((command - desired) ** 2, axis=0)))}",
        f"max abs command wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(command - desired), axis=0))}",
        f"rms applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.sqrt(np.mean((applied - desired) ** 2, axis=0)))}",
        f"max abs applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(applied - desired), axis=0))}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_position(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["time"])
    position = np.asarray(result["position"])
    desired = np.asarray(result["position_desired"])
    error = np.asarray(result["position_error"])
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    labels = ("x", "y", "z")
    for i, label in enumerate(labels):
        axes[i].plot(time, desired[:, i], "k--", linewidth=1.0, label=f"desired {label}")
        axes[i].plot(time, position[:, i], linewidth=0.9, label=f"actual {label}")
        axes[i].set_ylabel(f"{label} [m]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[3].plot(time, np.linalg.norm(error, axis=1), linewidth=1.0, label="position error norm")
    axes[3].set_ylabel("error [m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "position_tracking.png", dpi=180)
    plt.close(fig)


def plot_velocity(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["time"])
    velocity = np.asarray(result["velocity"])
    desired = np.asarray(result["velocity_desired"])
    speed = np.linalg.norm(velocity, axis=1)
    desired_speed = np.linalg.norm(desired, axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, label in enumerate(("vx", "vy", "vz")):
        axes[i].plot(time, desired[:, i], "k--", linewidth=1.0, label=f"desired {label}")
        axes[i].plot(time, velocity[:, i], linewidth=0.9, label=f"actual {label}")
        axes[i].set_ylabel(f"{label} [m/s]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[3].plot(time, desired_speed, "k--", linewidth=1.0, label="desired speed")
    axes[3].plot(time, speed, linewidth=0.9, label="actual speed")
    axes[3].axhline(6.0, color="tab:red", linewidth=0.9, linestyle=":", label="6 m/s")
    axes[3].set_ylabel("|v| [m/s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_tracking.png", dpi=180)
    plt.close(fig)


def plot_attitude(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["time"])
    rpy = np.unwrap(np.asarray(result["rpy"]), axis=0)
    desired = np.unwrap(np.asarray(result["rpy_desired"]), axis=0)
    attitude_error = np.asarray(result["attitude_error"])
    omega_error = np.asarray(result["omega_error"])
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    for i, label in enumerate(("roll", "pitch", "yaw")):
        axes[i].plot(time, desired[:, i], "k--", linewidth=1.0, label=f"desired {label}")
        axes[i].plot(time, rpy[:, i], linewidth=0.9, label=f"actual {label}")
        axes[i].set_ylabel(f"{label} [rad]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[3].plot(time, np.linalg.norm(attitude_error, axis=1), linewidth=1.0, label="attitude error norm")
    axes[3].set_ylabel("e_R")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[4].plot(time, np.linalg.norm(omega_error, axis=1), linewidth=1.0, label="omega error norm")
    axes[4].set_ylabel("e_omega [rad/s]")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "attitude_tracking.png", dpi=180)
    plt.close(fig)


def plot_wrench(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["time"])
    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    for i, (axis, label) in enumerate(zip(axes, WRENCH_NAMES)):
        axis.plot(time, desired[:, i], "k--", linewidth=1.0, label=f"desired {label}")
        axis.plot(time, command[:, i], linewidth=0.9, label=f"allocated {label}")
        axis.plot(time, applied[:, i], linewidth=0.9, label=f"applied {label}")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "wrench_tracking.png", dpi=180)
    plt.close(fig)


def plot_motors(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["time"])
    command_force = np.asarray(result["command_motor_force"])
    applied_force = np.asarray(result["applied_motor_force"])
    command_omega = np.asarray(result["command_motor_omega"])
    actual_omega = np.asarray(result["actual_motor_omega"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(command_force.shape[1]):
        axes[0].plot(time, command_force[:, i], linewidth=0.9, label=f"cmd f{i + 1}")
        axes[0].plot(time, applied_force[:, i], "--", linewidth=0.9, label=f"applied f{i + 1}")
        axes[1].plot(time, command_omega[:, i], linewidth=0.9, label=f"cmd omega{i + 1}")
        axes[1].plot(time, actual_omega[:, i], "--", linewidth=0.9, label=f"actual omega{i + 1}")
    axes[0].set_ylabel("motor force [N]")
    axes[1].set_ylabel("motor omega [rad/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "motor_forces_and_speeds.png", dpi=180)
    plt.close(fig)


def plot_trajectory(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    position = np.asarray(result["position"])
    desired = np.asarray(result["position_desired"])
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(desired[:, 0], desired[:, 1], desired[:, 2], "k--", linewidth=1.0, label="desired")
    axis.plot(position[:, 0], position[:, 1], position[:, 2], linewidth=1.0, label="actual")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="upper right")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_3d.png", dpi=180)
    plt.close(fig)


def save_results(output_dir: Path, result: dict[str, np.ndarray | str | float], args: argparse.Namespace) -> None:
    save_matrix(
        output_dir / "allocation_matrix_force_to_wrench.txt",
        np.asarray(result["allocation"]),
        "A_force rows [Fx Fy Fz Mx My Mz], columns motor forces [N]",
    )
    save_matrix(
        output_dir / "allocation_matrix_omega_squared_to_wrench.txt",
        np.asarray(result["omega_squared_allocation"]),
        "A_omega2 rows [Fx Fy Fz Mx My Mz], columns omega_i^2 [(rad/s)^2]",
    )
    write_summary(output_dir, result, args)
    plot_position(output_dir, result)
    plot_velocity(output_dir, result)
    plot_attitude(output_dir, result)
    plot_wrench(output_dir, result)
    plot_motors(output_dir, result)
    plot_trajectory(output_dir, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 6-DOF geometric pose tracking for the tilted hexrotor.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--actuator-model",
        choices=("plugin", "force"),
        default="plugin",
        help="Use real RotorMotor plugin actuators or native force actuators with Python-side speed dynamics.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo's native Python passive viewer while this script runs the controller.",
    )
    parser.add_argument(
        "--viewer-speed",
        type=float,
        default=1.0,
        help="Viewer playback speed multiplier. 1.0 is real time.",
    )
    parser.add_argument(
        "--viewer-sync-interval",
        type=int,
        default=5,
        help="Synchronize the viewer every N simulation steps.",
    )
    parser.add_argument(
        "--viewer-no-realtime",
        dest="viewer_realtime",
        action="store_false",
        help="Run as fast as possible while still updating the viewer.",
    )
    parser.add_argument(
        "--viewer-hold",
        action="store_true",
        help="Keep the viewer open at the final state until the window is closed.",
    )
    parser.add_argument("--viewer-distance", type=float, default=8.0, help="Tracking camera distance [m].")
    parser.add_argument("--viewer-azimuth", type=float, default=135.0, help="Tracking camera azimuth [deg].")
    parser.add_argument("--viewer-elevation", type=float, default=-20.0, help="Tracking camera elevation [deg].")
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--mass", type=float, default=0.85)
    parser.add_argument("--inertia", nargs=3, type=float, default=(0.01, 0.01, 0.02))
    parser.add_argument(
        "--joint-type",
        choices=("free", "ball"),
        default="free",
        help="Use ball for attitude tuning, free for full translational pose tracking.",
    )
    parser.add_argument("--initial-height", type=float, default=3.0)
    parser.add_argument("--initial-position-offset", nargs=3, type=float, default=(0.03, -0.02, 0.02))
    parser.add_argument("--initial-rpy-offset", nargs=3, type=float, default=(0.03, -0.02, 0.04))
    parser.add_argument("--radius", type=float, default=0.16)
    parser.add_argument("--tilt-deg", type=float, default=35.0)
    parser.add_argument("--axis-offset-deg", type=float, default=30.0)
    parser.add_argument("--km-over-kf", type=float, default=0.015)
    parser.add_argument("--kf", type=float, default=DEFAULT_KF)
    parser.add_argument("--omega-max", type=float, default=3000.0)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--delay", type=float, default=0.018)
    parser.add_argument("--nsample", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--allocator",
        choices=("osqp", "scipy"),
        default="osqp",
        help="Bounded motor-force allocation solver.",
    )
    parser.add_argument(
        "--allocation-regularization",
        type=float,
        default=1.0e-9,
        help="Small force regularization rho in the allocation QP.",
    )
    parser.add_argument("--osqp-eps-abs", type=float, default=1.0e-7, help="OSQP absolute tolerance.")
    parser.add_argument("--osqp-eps-rel", type=float, default=1.0e-7, help="OSQP relative tolerance.")
    parser.add_argument("--osqp-max-iter", type=int, default=4000, help="OSQP maximum iterations.")
    parser.add_argument(
        "--osqp-polish",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable OSQP polishing.",
    )
    parser.add_argument("--osqp-verbose", action="store_true", help="Print OSQP solver output.")
    parser.add_argument("--floor-size", type=float, default=30.0, help="Rendered checker-floor half-size [m].")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--kp-position", nargs=3, type=float, default=(2.0, 2.0, 3.0))
    parser.add_argument("--kd-position", nargs=3, type=float, default=(2.5, 2.5, 3.0))
    parser.add_argument("--kr-attitude", nargs=3, type=float, default=(0.70, 0.70, 0.95))
    parser.add_argument("--kw-attitude", nargs=3, type=float, default=(0.18, 0.18, 0.24))
    parser.add_argument("--allocation-force-weights", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--allocation-moment-weights", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--reference-position-center", nargs=3, type=float, default=(0.0, 0.0, 3.0))
    parser.add_argument("--reference-position-amplitude", nargs=3, type=float, default=(11.5, 4.0, 0.30))
    parser.add_argument("--reference-position-frequency", nargs=3, type=float, default=(0.04, 0.03, 0.04))
    parser.add_argument("--reference-position-phase", nargs=3, type=float, default=(0.0, 0.0, 0.4))
    parser.add_argument("--reference-attitude-amplitude", nargs=3, type=float, default=(0.08, 0.06, 0.18))
    parser.add_argument("--reference-attitude-frequency", nargs=3, type=float, default=(0.06, 0.05, 0.04))
    parser.add_argument("--reference-attitude-phase", nargs=3, type=float, default=(0.3, 1.1, 0.2))
    parser.add_argument("--metrics-start-time", type=float, default=2.0)
    parser.add_argument("--orientation-diff-step", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tracking(args)
    output_dir = args.output_dir.resolve()
    save_results(output_dir, result, args)

    sim_time = np.asarray(result["time"])
    position_error = np.asarray(result["position_error"])
    attitude_error = np.asarray(result["attitude_error"])
    metrics_mask = sim_time >= args.metrics_start_time
    if not np.any(metrics_mask):
        metrics_mask = np.ones_like(sim_time, dtype=bool)
    speed = np.linalg.norm(np.asarray(result["velocity"]), axis=1)
    desired_speed = np.linalg.norm(np.asarray(result["velocity_desired"]), axis=1)
    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    residual = np.asarray(result["allocation_residual"])
    print("Omnidirectional hexrotor geometric tracking complete")
    print(f"output: {output_dir}")
    print(f"model: {result['model_path']}")
    print(f"joint type: {result['joint_type']}")
    print(f"gravity z: {float(result['gravity_z']):.6g} m/s^2")
    print(f"allocator: {result['allocator']}")
    print(f"allocator statuses: {result['allocator_status_counts']}")
    print(f"allocation rank: {int(result['rank'])}")
    print(f"allocation condition number: {float(result['condition_number']):.6g}")
    print(f"max desired speed: {float(np.max(desired_speed)):.6g} m/s")
    print(f"max actual speed: {float(np.max(speed)):.6g} m/s")
    print(f"max position error: {float(np.max(np.linalg.norm(position_error, axis=1))):.6g} m")
    print(f"rms position error: {float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))):.6g} m")
    print(f"max attitude error norm: {float(np.max(np.linalg.norm(attitude_error, axis=1))):.6g}")
    print(
        f"max attitude error norm after {args.metrics_start_time:.6g} s: "
        f"{float(np.max(np.linalg.norm(attitude_error[metrics_mask], axis=1))):.6g}"
    )
    print(
        "max abs command wrench error [Fx, Fy, Fz, Mx, My, Mz]: "
        f"{_fmt(np.max(np.abs(command - desired), axis=0))}"
    )
    print(
        "max abs moment allocation residual [Mx, My, Mz]: "
        f"{_fmt(np.max(np.abs(residual[:, 3:6]), axis=0))}"
    )
    print(
        "max abs applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: "
        f"{_fmt(np.max(np.abs(applied - desired), axis=0))}"
    )
    print(f"summary: {output_dir / 'summary.txt'}")
    print(f"csv: {result['csv_path']}")
    print(
        "plots: "
        f"{output_dir / 'position_tracking.png'}, "
        f"{output_dir / 'attitude_tracking.png'}, "
        f"{output_dir / 'wrench_tracking.png'}"
    )


if __name__ == "__main__":
    main()
