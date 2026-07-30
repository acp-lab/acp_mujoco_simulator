#!/usr/bin/env python3
"""Validate a tilted 6-motor omnidirectional rotor allocation.

The generated MuJoCo model follows the quadrotor_verify_mass_inertia.xml style:
one free body, visible body/arm/rotor geoms, a checker floor, lighting, camera,
six rotor sites, and six site actuators. By default the six actuators are real
MujocoRosUtils::RotorMotor plugin actuators.

The script reads the per-unit actuator wrench from MuJoCo, builds

    wrench_body = A_force @ f_motor

and combines it with the motor thrust law

    f_i = kf * omega_i^2

to get

    wrench_body = A_force @ diag(kf_i) @ omega_squared.

It then solves bounded motor forces for a time-varying desired body wrench
[Fx, Fy, Fz, Mx, My, Mz], converts those forces to motor speed commands, and
checks the applied MuJoCo wrench from data.actuator_force.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from compare_quadrotor_actuator_wrenches import actuator_matrix, load_actuator_wrenches


THIS_DIR = Path(__file__).resolve().parent
WS_DIR = THIS_DIR.parents[2]
DEFAULT_OUTPUT_DIR = THIS_DIR / "omnidirectional_hexrotor_wrench_mapping_results"
DEFAULT_PLUGIN = (
    WS_DIR / "mujoco-3.10.0" / "bin" / "mujoco_plugin" / "libMujocoRosUtilsPlugin.so"
)
DEFAULT_KF = 2.063751543951938e-06
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_PLUGIN_LOADED = False


@dataclass(frozen=True)
class Geometry:
    positions: np.ndarray
    axes: np.ndarray
    yaw_signs: np.ndarray
    km_over_kf: float


class ScipyBoundedLeastSquaresAllocator:
    def __init__(
        self,
        allocation: np.ndarray,
        force_min: np.ndarray,
        force_max: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        self.name = "scipy"
        self.allocation = np.asarray(allocation, dtype=float)
        self.force_min = np.asarray(force_min, dtype=float)
        self.force_max = np.asarray(force_max, dtype=float)
        self.weights = _allocator_weights(self.allocation, weights)
        self.weighted_allocation = self.weights[:, None] * self.allocation

    def solve(self, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        desired = np.asarray(desired, dtype=float)
        weighted_desired = self.weights * desired
        try:
            from scipy.optimize import lsq_linear

            result = lsq_linear(
                self.weighted_allocation,
                weighted_desired,
                bounds=(self.force_min, self.force_max),
                lsmr_tol="auto",
            )
            forces = result.x
            status = f"scipy_lsq_linear_status_{result.status}"
        except Exception:
            forces, *_ = np.linalg.lstsq(
                self.weighted_allocation, weighted_desired, rcond=None
            )
            forces = np.clip(forces, self.force_min, self.force_max)
            status = "numpy_lstsq_clipped"
        return forces, self.allocation @ forces - desired, status


class OsqpBoundedLeastSquaresAllocator:
    def __init__(
        self,
        allocation: np.ndarray,
        force_min: np.ndarray,
        force_max: np.ndarray,
        weights: np.ndarray | None = None,
        regularization: float = 1.0e-9,
        eps_abs: float = 1.0e-7,
        eps_rel: float = 1.0e-7,
        max_iter: int = 4000,
        polish: bool = False,
        verbose: bool = False,
    ) -> None:
        import osqp
        import scipy.sparse as sp

        self.name = "osqp"
        self.allocation = np.asarray(allocation, dtype=float)
        self.force_min = np.asarray(force_min, dtype=float)
        self.force_max = np.asarray(force_max, dtype=float)
        self.weights = _allocator_weights(self.allocation, weights)
        self.weighted_allocation = self.weights[:, None] * self.allocation

        nu = self.allocation.shape[1]
        p_dense = self.weighted_allocation.T @ self.weighted_allocation
        p_dense += float(regularization) * np.eye(nu)
        p_dense = 0.5 * (p_dense + p_dense.T)
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sp.csc_matrix(np.triu(p_dense)),
            q=np.zeros(nu),
            A=sp.eye(nu, format="csc"),
            l=self.force_min,
            u=self.force_max,
            verbose=verbose,
            polish=polish,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
        )

    def solve(self, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        desired = np.asarray(desired, dtype=float)
        weighted_desired = self.weights * desired
        q = -(self.weighted_allocation.T @ weighted_desired)
        self.solver.update(q=q)
        result = self.solver.solve()
        status = getattr(result.info, "status", "unknown")
        if result.x is None or "solved" not in status.lower():
            raise RuntimeError(f"OSQP allocation failed with status: {status}")
        forces = np.clip(
            np.asarray(result.x, dtype=float), self.force_min, self.force_max
        )
        return forces, self.allocation @ forces - desired, f"osqp_{status}"


def _allocator_weights(
    allocation: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    if weights is None:
        return np.ones(allocation.shape[0], dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.shape != (allocation.shape[0],):
        raise ValueError(
            f"Expected {allocation.shape[0]} allocation weights, got shape {values.shape}."
        )
    return values


def make_bounded_motor_force_allocator(
    allocation: np.ndarray,
    force_min: np.ndarray,
    force_max: np.ndarray,
    weights: np.ndarray | None = None,
    allocator: str = "osqp",
    regularization: float = 1.0e-9,
    osqp_eps_abs: float = 1.0e-7,
    osqp_eps_rel: float = 1.0e-7,
    osqp_max_iter: int = 4000,
    osqp_polish: bool = False,
    osqp_verbose: bool = False,
    fallback_to_scipy: bool = True,
) -> OsqpBoundedLeastSquaresAllocator | ScipyBoundedLeastSquaresAllocator:
    if allocator == "osqp":
        try:
            return OsqpBoundedLeastSquaresAllocator(
                allocation,
                force_min,
                force_max,
                weights=weights,
                regularization=regularization,
                eps_abs=osqp_eps_abs,
                eps_rel=osqp_eps_rel,
                max_iter=osqp_max_iter,
                polish=osqp_polish,
                verbose=osqp_verbose,
            )
        except Exception as exc:
            if not fallback_to_scipy:
                raise
            print(
                f"OSQP allocator unavailable ({exc}); falling back to SciPy bounded least squares."
            )
    elif allocator != "scipy":
        raise ValueError(f"Unsupported allocator: {allocator}")
    return ScipyBoundedLeastSquaresAllocator(
        allocation, force_min, force_max, weights=weights
    )


def preload_allocator_backend(allocator: str) -> None:
    if allocator == "osqp":
        import osqp  # noqa: F401
        import scipy.sparse  # noqa: F401


def _fmt(values: np.ndarray) -> str:
    return (
        "["
        + ", ".join(f"{value:.6g}" for value in np.asarray(values, dtype=float))
        + "]"
    )


def load_plugin(plugin_library: Path) -> None:
    global _PLUGIN_LOADED
    if _PLUGIN_LOADED:
        return
    if not plugin_library.exists():
        raise FileNotFoundError(
            f"Plugin library not found: {plugin_library}\n"
            "Build MujocoRosUtils against the selected MuJoCo version before running plugin validation."
        )
    workspace_dir = THIS_DIR.parents[2]
    preload_libraries = (
        workspace_dir
        / "install"
        / "mujoco_ros_utils"
        / "lib"
        / "libmujoco_ros_utils__rosidl_typesupport_cpp.so",
        workspace_dir
        / "install"
        / "quadrotor_msgs"
        / "lib"
        / "libquadrotor_msgs__rosidl_typesupport_cpp.so",
    )
    for library in preload_libraries:
        if library.exists():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    sibling_core_library = plugin_library.with_name("libMujocoRosUtils.so")
    if sibling_core_library.exists():
        ctypes.CDLL(str(sibling_core_library), mode=ctypes.RTLD_GLOBAL)
    mujoco.mj_loadPluginLibrary(str(plugin_library))
    _PLUGIN_LOADED = True


def activation_by_actuator(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for actuator_id in range(model.nu):
        act_adr = int(model.actuator_actadr[actuator_id])
        act_num = int(model.actuator_actnum[actuator_id])
        if act_adr >= 0 and act_num > 0:
            values.append(float(data.act[act_adr]))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def set_activation_by_actuator(
    model: mujoco.MjModel, data: mujoco.MjData, values: np.ndarray
) -> None:
    for actuator_id, value in enumerate(np.asarray(values, dtype=float)):
        act_adr = int(model.actuator_actadr[actuator_id])
        act_num = int(model.actuator_actnum[actuator_id])
        if act_adr >= 0 and act_num > 0:
            data.act[act_adr] = value


def make_tilted_hex_geometry(
    radius: float,
    tilt_deg: float,
    axis_offset_deg: float,
    km_over_kf: float,
) -> Geometry:
    tilt = np.deg2rad(tilt_deg)
    axis_offset = np.deg2rad(axis_offset_deg)
    positions = []
    axes = []
    yaw_signs = []
    for i in range(6):
        theta = 2.0 * np.pi * i / 6.0
        positions.append([radius * np.cos(theta), radius * np.sin(theta), 0.0])

        # Alternating azimuth offsets avoid the rank loss of pure radial or
        # pure tangential tilts.
        axis_azimuth = theta + ((-1.0) ** i) * axis_offset
        axis = np.array(
            [
                np.sin(tilt) * np.cos(axis_azimuth),
                np.sin(tilt) * np.sin(axis_azimuth),
                np.cos(tilt),
            ],
            dtype=float,
        )
        axes.append(axis / np.linalg.norm(axis))
        yaw_signs.append(1.0 if i % 2 == 0 else -1.0)
    return Geometry(
        positions=np.asarray(positions, dtype=float),
        axes=np.asarray(axes, dtype=float),
        yaw_signs=np.asarray(yaw_signs, dtype=float),
        km_over_kf=float(km_over_kf),
    )


def generated_xml(args: argparse.Namespace, geometry: Geometry) -> str:
    site_lines = []
    geom_lines = []
    actuator_lines = []
    force_max = args.kf * args.omega_max * args.omega_max
    gravity_z = getattr(args, "gravity_z", 0.0)
    floor_size = getattr(args, "floor_size", 10.0)
    floor_repeat = max(1.0, floor_size / 2.0)
    joint_type = getattr(args, "joint_type", "free")
    if joint_type == "free":
        joint_line = '<joint name="omni_1" type="free" damping="0.001"/>'
    elif joint_type == "ball":
        joint_line = '<joint name="omni_1" type="ball" damping="0.001"/>'
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")
    for i, (position, axis, yaw_sign) in enumerate(
        zip(geometry.positions, geometry.axes, geometry.yaw_signs), start=1
    ):
        site_lines.append(
            f"""
      <site name="omni_1_rotor{i}_site"
            type="sphere"
            pos="{position[0]:.16g} {position[1]:.16g} {position[2]:.16g}"
            zaxis="{axis[0]:.16g} {axis[1]:.16g} {axis[2]:.16g}"
            size="0.006"
            rgba="0 1 0 1"/>"""
        )
        geom_lines.append(
            f"""
      <geom name="omni_1_arm{i}"
            type="capsule"
            fromto="0 0 0 {position[0]:.16g} {position[1]:.16g} {position[2]:.16g}"
            size="0.009"
            rgba="1 0.5 0 1"/>
      <geom name="omni_1_rotor{i}_marker"
            type="cylinder"
            pos="{position[0]:.16g} {position[1]:.16g} 0.0"
            size="0.035 0.003"
            zaxis="{axis[0]:.16g} {axis[1]:.16g} {axis[2]:.16g}"
            rgba="1 0 0 0.8"/>"""
        )
        if args.actuator_model == "plugin":
            actuator_lines.append(
                f"""
    <plugin name="omni_1_rotor{i}"
            site="omni_1_rotor{i}_site"
            plugin="MujocoRosUtils::RotorMotor"
            actdim="1"
            ctrlrange="0 {args.omega_max:.16g}"
            gear="0 0 1 0 0 {yaw_sign * geometry.km_over_kf:.16g}"
            nsample="{args.nsample}"
            delay="{args.delay:.16g}"
            interp="zoh">
      <config key="kf" value="{args.kf:.16g}"/>
      <config key="tau" value="{args.tau:.16g}"/>
    </plugin>"""
            )
        else:
            actuator_lines.append(
                f"""
    <motor name="omni_1_rotor{i}"
           site="omni_1_rotor{i}_site"
           gear="0 0 1 0 0 {yaw_sign * geometry.km_over_kf:.16g}"
           ctrllimited="true"
           ctrlrange="0 {force_max:.16g}"/>"""
            )

    extension = ""
    if args.actuator_model == "plugin":
        extension = """
  <extension>
    <plugin plugin="MujocoRosUtils::RotorMotor"/>
  </extension>"""

    return f"""
<mujoco model="omnidirectional_hexrotor_wrench_validation">
  <compiler angle="radian"/>
  <option timestep="{args.dt:.16g}" gravity="0 0 {gravity_z:.16g}" integrator="RK4">
    <flag energy="enable" contact="enable"/>
  </option>
  {extension}
  <asset>
    <texture name="texchecker" type="2d" builtin="checker" width="512" height="512" rgb1="0.8 0.8 0.8" rgb2="0.2 0.2 0.2"/>
    <material name="matchecker" texture="texchecker" texrepeat="{floor_repeat:.16g} {floor_repeat:.16g}" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 {max(2.5, floor_size * 0.25):.16g}" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="{floor_size:.16g} {floor_size:.16g} 0.1" material="matchecker" condim="3"/>
    <camera name="overview" pos="0 -1.35 2.25" xyaxes="1 0 0 0 0.86 0.51"/>
    <body name="omni_1" pos="0 0 {args.initial_height:.16g}">
      {joint_line}
      <inertial pos="0 0 0" mass="{args.mass:.16g}" diaginertia="{args.inertia[0]:.16g} {args.inertia[1]:.16g} {args.inertia[2]:.16g}"/>
      <geom name="omni_1_core" type="box" size="0.055 0.055 0.022" rgba="1 1 0 1"/>
      {"".join(geom_lines)}
      {"".join(site_lines)}
      <site name="omni_1_imu" pos="0 0 0" size="0.006" rgba="0 1 1 1"/>
      <camera name="onboard" pos="0.06 0 -0.025" euler="0 0 -1.57079632679"/>
    </body>
  </worldbody>
  <actuator>
    {"".join(actuator_lines)}
  </actuator>
</mujoco>
"""


def solve_bounded_motor_forces(
    allocation: np.ndarray,
    desired: np.ndarray,
    force_min: np.ndarray,
    force_max: np.ndarray,
    weights: np.ndarray | None = None,
    allocator: str = "osqp",
    regularization: float = 1.0e-9,
    osqp_eps_abs: float = 1.0e-7,
    osqp_eps_rel: float = 1.0e-7,
    osqp_max_iter: int = 4000,
    osqp_polish: bool = True,
    osqp_verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    motor_allocator = make_bounded_motor_force_allocator(
        allocation,
        force_min,
        force_max,
        weights=weights,
        allocator=allocator,
        regularization=regularization,
        osqp_eps_abs=osqp_eps_abs,
        osqp_eps_rel=osqp_eps_rel,
        osqp_max_iter=osqp_max_iter,
        osqp_polish=osqp_polish,
        osqp_verbose=osqp_verbose,
    )
    return motor_allocator.solve(desired)


def desired_wrench(
    time: float,
    nominal: np.ndarray,
    amplitude: np.ndarray,
    frequency: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    return nominal + amplitude * np.sin(2.0 * np.pi * frequency * time + phase)


def delayed_command(
    current_time: float,
    delay: float,
    command_history: list[tuple[float, np.ndarray]],
    fallback: np.ndarray,
) -> np.ndarray:
    if delay <= 0.0:
        return command_history[-1][1]
    query_time = current_time - delay
    selected = fallback
    for sample_time, command in reversed(command_history):
        if sample_time <= query_time:
            selected = command
            break
    return selected


def free_joint_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "omni_1")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    return data.qpos[qpos_adr : qpos_adr + 7].copy(), data.qvel[
        qvel_adr : qvel_adr + 6
    ].copy()


def run_validation(args: argparse.Namespace) -> dict[str, np.ndarray | str | float]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preload_allocator_backend(args.allocator)
    if args.actuator_model == "plugin":
        if args.tau <= 0.0:
            raise ValueError("RotorMotor plugin mode requires --tau > 0.")
        load_plugin(args.plugin_library)

    geometry = make_tilted_hex_geometry(
        args.radius, args.tilt_deg, args.axis_offset_deg, args.km_over_kf
    )
    xml_text = generated_xml(args, geometry)
    model_path = output_dir / "omnidirectional_hexrotor_validation.xml"
    model_path.write_text(xml_text, encoding="utf-8")

    model, wrenches = load_actuator_wrenches(model_path, "omni_1")
    model.opt.gravity[:] = (0.0, 0.0, args.gravity_z)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    allocation = actuator_matrix(wrenches)
    kf_vector = np.full(model.nu, args.kf, dtype=float)
    omega_squared_allocation = allocation @ np.diag(kf_vector)
    force_max_scalar = args.kf * args.omega_max * args.omega_max
    force_min = np.zeros(model.nu)
    force_max = np.full(model.nu, force_max_scalar)
    allocator = make_bounded_motor_force_allocator(
        allocation,
        force_min,
        force_max,
        allocator=args.allocator,
        regularization=args.allocation_regularization,
        osqp_eps_abs=args.osqp_eps_abs,
        osqp_eps_rel=args.osqp_eps_rel,
        osqp_max_iter=args.osqp_max_iter,
        osqp_polish=args.osqp_polish,
        osqp_verbose=args.osqp_verbose,
    )

    nominal_forces = np.full(model.nu, args.nominal_motor_force, dtype=float)
    nominal_wrench = allocation @ nominal_forces
    amplitude = np.asarray(args.wrench_amplitude, dtype=float)
    frequency = np.asarray(args.wrench_frequency, dtype=float)
    phase = np.asarray(args.wrench_phase, dtype=float)

    initial_desired = desired_wrench(0.0, nominal_wrench, amplitude, frequency, phase)
    initial_force_cmd, _, _ = allocator.solve(initial_desired)
    omega_actual = np.sqrt(np.maximum(initial_force_cmd, 0.0) / args.kf)
    command_history: list[tuple[float, np.ndarray]] = [(0.0, omega_actual.copy())]
    if args.actuator_model == "plugin":
        set_activation_by_actuator(model, data, omega_actual)
        data.ctrl[:] = omega_actual
        mujoco.mj_forward(model, data)

    nsteps = int(np.ceil(args.duration / model.opt.timestep))
    csv_path = output_dir / "omnidirectional_hexrotor_wrench_mapping.csv"

    times = []
    desired_log = []
    command_force_log = []
    command_omega_log = []
    actual_omega_log = []
    applied_force_log = []
    command_wrench_log = []
    applied_wrench_log = []
    residual_log = []
    qpos_log = []
    qvel_log = []
    allocator_status_counts: dict[str, int] = {}

    header = ["step", "time"]
    header += [f"desired_{name}" for name in WRENCH_NAMES]
    header += [f"command_wrench_{name}" for name in WRENCH_NAMES]
    header += [f"applied_wrench_{name}" for name in WRENCH_NAMES]
    header += [f"allocation_residual_{name}" for name in WRENCH_NAMES]
    header += [f"motor_force_cmd_{i + 1}" for i in range(model.nu)]
    header += [f"omega_cmd_{i + 1}" for i in range(model.nu)]
    header += [f"omega_actual_{i + 1}" for i in range(model.nu)]
    header += [f"motor_force_applied_{i + 1}" for i in range(model.nu)]
    header += ["qpos_x", "qpos_y", "qpos_z", "qpos_qw", "qpos_qx", "qpos_qy", "qpos_qz"]
    header += ["qvel_x", "qvel_y", "qvel_z", "qvel_wx", "qvel_wy", "qvel_wz"]

    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)

        for step in range(nsteps + 1):
            time = float(data.time)
            desired = desired_wrench(time, nominal_wrench, amplitude, frequency, phase)
            force_cmd, residual, allocator_status = allocator.solve(desired)
            allocator_status_counts[allocator_status] = (
                allocator_status_counts.get(allocator_status, 0) + 1
            )
            omega_cmd = np.sqrt(np.maximum(force_cmd, 0.0) / args.kf)

            if args.actuator_model == "plugin":
                data.ctrl[:] = omega_cmd
            else:
                command_history.append((time, omega_cmd.copy()))
                delayed_omega_cmd = delayed_command(
                    time, args.delay, command_history, command_history[0][1]
                )

                if args.tau <= 0.0:
                    omega_actual = delayed_omega_cmd.copy()
                else:
                    dt = float(model.opt.timestep)
                    filter_alpha = np.exp(-dt / args.tau)
                    omega_actual = (
                        filter_alpha * omega_actual
                        + (1.0 - filter_alpha) * delayed_omega_cmd
                    )

                applied_force = args.kf * omega_actual * omega_actual
                data.ctrl[:] = applied_force
            mujoco.mj_forward(model, data)

            actuator_force = data.actuator_force.copy()
            if args.actuator_model == "plugin":
                omega_actual = activation_by_actuator(model, data)
            command_wrench = allocation @ force_cmd
            applied_wrench = allocation @ actuator_force
            qpos, qvel = free_joint_state(model, data)

            writer.writerow(
                [step, time]
                + desired.tolist()
                + command_wrench.tolist()
                + applied_wrench.tolist()
                + residual.tolist()
                + force_cmd.tolist()
                + omega_cmd.tolist()
                + omega_actual.tolist()
                + actuator_force.tolist()
                + qpos.tolist()
                + qvel.tolist()
            )

            times.append(time)
            desired_log.append(desired)
            command_force_log.append(force_cmd)
            command_omega_log.append(omega_cmd)
            actual_omega_log.append(omega_actual.copy())
            applied_force_log.append(actuator_force)
            command_wrench_log.append(command_wrench)
            applied_wrench_log.append(applied_wrench)
            residual_log.append(residual)
            qpos_log.append(qpos)
            qvel_log.append(qvel)

            if step < nsteps:
                mujoco.mj_step(model, data)

    return {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "allocation": allocation,
        "omega_squared_allocation": omega_squared_allocation,
        "singular_values": np.linalg.svd(allocation, compute_uv=False),
        "condition_number": float(np.linalg.cond(allocation)),
        "rank": int(np.linalg.matrix_rank(allocation)),
        "nominal_wrench": nominal_wrench,
        "times": np.asarray(times),
        "desired_wrench": np.asarray(desired_log),
        "command_wrench": np.asarray(command_wrench_log),
        "applied_wrench": np.asarray(applied_wrench_log),
        "allocation_residual": np.asarray(residual_log),
        "command_motor_force": np.asarray(command_force_log),
        "command_motor_omega": np.asarray(command_omega_log),
        "actual_motor_omega": np.asarray(actual_omega_log),
        "applied_motor_force": np.asarray(applied_force_log),
        "qpos": np.asarray(qpos_log),
        "qvel": np.asarray(qvel_log),
        "force_max": force_max_scalar,
        "omega_max": args.omega_max,
        "kf": args.kf,
        "tau": args.tau,
        "delay": args.delay,
        "allocator": allocator.name,
        "allocation_regularization": args.allocation_regularization,
        "allocator_status_counts": allocator_status_counts,
        "actuator_model": args.actuator_model,
        "plugin_library": str(args.plugin_library),
        "wrench_amplitude": amplitude,
        "wrench_frequency": frequency,
        "wrench_phase": phase,
    }


def save_matrix(path: Path, matrix: np.ndarray, header: str) -> None:
    np.savetxt(path, matrix, fmt="%.12g", header=header)


def write_summary(
    output_dir: Path, result: dict[str, np.ndarray | str | float]
) -> None:
    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    allocation_residual = np.asarray(result["allocation_residual"])
    command_force = np.asarray(result["command_motor_force"])
    command_omega = np.asarray(result["command_motor_omega"])
    actual_omega = np.asarray(result["actual_motor_omega"])
    applied_force = np.asarray(result["applied_motor_force"])

    lines = [
        "Omnidirectional hexrotor wrench mapping validation",
        "",
        f"model xml: {result['model_path']}",
        f"csv: {result['csv_path']}",
        f"actuator model: {result['actuator_model']}",
        f"plugin library: {result['plugin_library']}",
        f"kf: {float(result['kf']):.16g} N/(rad/s)^2",
        f"omega max: {float(result['omega_max']):.6g} rad/s",
        f"motor force max: {float(result['force_max']):.6g} N",
        f"motor speed tau: {float(result['tau']):.6g} s",
        f"motor speed delay: {float(result['delay']):.6g} s",
        f"allocator: {result['allocator']}",
        f"allocation regularization: {float(result['allocation_regularization']):.6g}",
        f"allocator status counts: {result['allocator_status_counts']}",
        "",
        f"allocation rank: {int(result['rank'])}",
        f"allocation condition number: {float(result['condition_number']):.6g}",
        f"allocation singular values: {_fmt(np.asarray(result['singular_values']))}",
        f"nominal wrench [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.asarray(result['nominal_wrench']))}",
        "",
        "Allocation matrix A_force, rows [Fx, Fy, Fz, Mx, My, Mz], columns motor forces:",
        np.array2string(
            np.asarray(result["allocation"]), precision=6, suppress_small=True
        ),
        "",
        "Direct omega^2 matrix A_omega2 = A_force @ diag(kf), rows [Fx, Fy, Fz, Mx, My, Mz]:",
        np.array2string(
            np.asarray(result["omega_squared_allocation"]),
            precision=12,
            suppress_small=False,
        ),
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
        f"max abs command allocation residual [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(allocation_residual), axis=0))}",
        f"rms command wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.sqrt(np.mean((command - desired) ** 2, axis=0)))}",
        f"max abs command wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(command - desired), axis=0))}",
        f"rms applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.sqrt(np.mean((applied - desired) ** 2, axis=0)))}",
        f"max abs applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.max(np.abs(applied - desired), axis=0))}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_wrench(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["times"])
    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    for i, (axis, label) in enumerate(zip(axes, WRENCH_NAMES)):
        axis.plot(time, desired[:, i], "k--", linewidth=1.0, label=f"desired {label}")
        axis.plot(
            time, command[:, i], linewidth=0.9, label=f"allocated command {label}"
        )
        axis.plot(time, applied[:, i], linewidth=0.9, label=f"applied {label}")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "wrench_tracking.png", dpi=180)
    plt.close(fig)


def plot_motors(output_dir: Path, result: dict[str, np.ndarray | str | float]) -> None:
    time = np.asarray(result["times"])
    command_force = np.asarray(result["command_motor_force"])
    applied_force = np.asarray(result["applied_motor_force"])
    command_omega = np.asarray(result["command_motor_omega"])
    actual_omega = np.asarray(result["actual_motor_omega"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i in range(command_force.shape[1]):
        axes[0].plot(time, command_force[:, i], linewidth=0.9, label=f"cmd f{i + 1}")
        axes[0].plot(
            time, applied_force[:, i], "--", linewidth=0.9, label=f"applied f{i + 1}"
        )
        axes[1].plot(
            time, command_omega[:, i], linewidth=0.9, label=f"cmd omega{i + 1}"
        )
        axes[1].plot(
            time, actual_omega[:, i], "--", linewidth=0.9, label=f"actual omega{i + 1}"
        )
    axes[0].set_ylabel("motor force [N]")
    axes[1].set_ylabel("motor omega [rad/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "motor_forces_and_speeds.png", dpi=180)
    plt.close(fig)


def plot_position(
    output_dir: Path, result: dict[str, np.ndarray | str | float]
) -> None:
    time = np.asarray(result["times"])
    qpos = np.asarray(result["qpos"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, label in enumerate(("x", "y", "z")):
        axes[i].plot(time, qpos[:, i], label=label)
        axes[i].set_ylabel(f"{label} [m]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "position_response.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a 6-motor omnidirectional wrench allocation."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--actuator-model",
        choices=("plugin", "force"),
        default="plugin",
        help="Use real RotorMotor plugin actuators or native force actuators with Python-side speed dynamics.",
    )
    parser.add_argument(
        "--plugin-library",
        type=Path,
        default=DEFAULT_PLUGIN,
        help=f"Path to libMujocoRosUtilsPlugin.so. Default: {DEFAULT_PLUGIN}",
    )
    parser.add_argument("--mass", type=float, default=0.85)
    parser.add_argument(
        "--inertia",
        nargs=3,
        type=float,
        default=(0.01, 0.01, 0.02),
        metavar=("IXX", "IYY", "IZZ"),
        help="Body diagonal inertia used by the generated visual validation model [kg m^2].",
    )
    parser.add_argument(
        "--initial-height",
        type=float,
        default=2.0,
        help="Initial body height in the viewer [m].",
    )
    parser.add_argument(
        "--joint-type",
        choices=("free", "ball"),
        default="free",
        help="Body joint in the generated model. Use ball for attitude-only test-stand validation.",
    )
    parser.add_argument(
        "--radius", type=float, default=0.16, help="Rotor radius from body origin [m]."
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=35.0,
        help="Rotor axis tilt away from body z [deg].",
    )
    parser.add_argument(
        "--axis-offset-deg",
        type=float,
        default=30.0,
        help="Alternating rotor-axis azimuth offset relative to radial direction [deg].",
    )
    parser.add_argument(
        "--km-over-kf",
        type=float,
        default=0.015,
        help="Rotor drag torque per thrust [m].",
    )
    parser.add_argument(
        "--kf", type=float, default=DEFAULT_KF, help="Thrust coefficient [N/(rad/s)^2]."
    )
    parser.add_argument(
        "--omega-max", type=float, default=3000.0, help="Maximum motor speed [rad/s]."
    )
    parser.add_argument(
        "--nominal-motor-force",
        type=float,
        default=2.0,
        help="Nominal force per motor [N].",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.025,
        help="First-order motor speed time constant [s].",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.018,
        help="Pure command delay for motor speed command [s].",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=8,
        help="MuJoCo actuator input-delay sample count.",
    )
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
    parser.add_argument(
        "--osqp-eps-abs", type=float, default=1.0e-7, help="OSQP absolute tolerance."
    )
    parser.add_argument(
        "--osqp-eps-rel", type=float, default=1.0e-7, help="OSQP relative tolerance."
    )
    parser.add_argument(
        "--osqp-max-iter", type=int, default=4000, help="OSQP maximum iterations."
    )
    parser.add_argument(
        "--osqp-polish",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable OSQP polishing.",
    )
    parser.add_argument(
        "--osqp-verbose", action="store_true", help="Print OSQP solver output."
    )
    parser.add_argument(
        "--floor-size",
        type=float,
        default=10.0,
        help="Rendered checker-floor half-size [m].",
    )
    parser.add_argument(
        "--gravity-z",
        type=float,
        default=0.0,
        help="World z gravity for the generated model [m/s^2]. Default keeps pure wrench validation gravity-free.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--wrench-amplitude",
        nargs=6,
        type=float,
        metavar=("FX", "FY", "FZ", "MX", "MY", "MZ"),
        default=(0.3, 0.3, 0.4, 0.025, 0.025, 0.015),
        help="Sine amplitude around nominal body wrench.",
    )
    parser.add_argument(
        "--wrench-frequency",
        nargs=6,
        type=float,
        metavar=("FX", "FY", "FZ", "MX", "MY", "MZ"),
        default=(0.31, 0.47, 0.23, 0.37, 0.41, 0.53),
        help="Sine frequencies for body wrench components [Hz].",
    )
    parser.add_argument(
        "--wrench-phase",
        nargs=6,
        type=float,
        metavar=("FX", "FY", "FZ", "MX", "MY", "MZ"),
        default=(0.0, 1.0, 2.0, 0.5, 1.5, 2.5),
        help="Sine phases for body wrench components [rad].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_validation(args)
    output_dir = args.output_dir.resolve()

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
    write_summary(output_dir, result)
    plot_wrench(output_dir, result)
    plot_motors(output_dir, result)
    plot_position(output_dir, result)

    desired = np.asarray(result["desired_wrench"])
    command = np.asarray(result["command_wrench"])
    applied = np.asarray(result["applied_wrench"])
    print("Omnidirectional hexrotor validation complete")
    print(f"output: {output_dir}")
    print(f"allocator: {result['allocator']}")
    print(f"allocator statuses: {result['allocator_status_counts']}")
    print(f"allocation rank: {int(result['rank'])}")
    print(f"allocation condition number: {float(result['condition_number']):.6g}")
    print(f"singular values: {_fmt(np.asarray(result['singular_values']))}")
    print(
        f"nominal wrench [Fx, Fy, Fz, Mx, My, Mz]: {_fmt(np.asarray(result['nominal_wrench']))}"
    )
    print(
        "max abs command wrench error [Fx, Fy, Fz, Mx, My, Mz]: "
        f"{_fmt(np.max(np.abs(command - desired), axis=0))}"
    )
    print(
        "max abs applied wrench error [Fx, Fy, Fz, Mx, My, Mz]: "
        f"{_fmt(np.max(np.abs(applied - desired), axis=0))}"
    )
    print(f"summary: {output_dir / 'summary.txt'}")
    print(
        f"plots: {output_dir / 'wrench_tracking.png'}, {output_dir / 'motor_forces_and_speeds.png'}"
    )


if __name__ == "__main__":
    main()
