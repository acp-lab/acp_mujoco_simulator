#!/usr/bin/env python3
"""Compare direct rotor-force actuators with the RotorMotor plugin.

The direct model receives desired motor thrusts

    data.ctrl[i] = f_cmd_i [N]

and uses MuJoCo's native activation dynamics to delay/filter thrust directly.

The plugin model receives desired rotor speeds

    data.ctrl[i] = omega_cmd_i [rad/s],  omega_cmd_i = sqrt(f_cmd_i / kf)

and the RotorMotor plugin stores actual rotor speed in data.act, then writes

    data.actuator_force[i] = kf * omega_i^2.

Therefore the correct comparison variable is data.actuator_force, not data.act.
For the direct force model data.act is filtered thrust; for the plugin model
data.act is rotor speed.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WS_DIR = THIS_DIR.parents[2]
DEFAULT_PLUGIN = WS_DIR / "mujoco-3.10.0" / "bin" / "mujoco_plugin" / "libMujocoRosUtilsPlugin.so"
DEFAULT_OUTPUT_DIR = THIS_DIR / "rotor_motor_force_comparison_results"

ROTOR_POSITIONS = np.asarray(
    (
        (0.1, 0.1, 0.0),
        (0.1, -0.1, 0.0),
        (-0.1, -0.1, 0.0),
        (-0.1, 0.1, 0.0),
    ),
    dtype=float,
)
YAW_SIGNS = np.asarray((1.0, -1.0, 1.0, -1.0), dtype=float)
WRENCH_LABELS = ("Fz", "Tx", "Ty", "Tz")
_PLUGIN_LOADED = False


def allocation_matrix(km_over_kf: float) -> np.ndarray:
    """Map motor thrusts [N] to [Fz, Tx, Ty, Tz] in the body frame."""
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, :] = 1.0
    matrix[1, :] = ROTOR_POSITIONS[:, 1]
    matrix[2, :] = -ROTOR_POSITIONS[:, 0]
    matrix[3, :] = YAW_SIGNS * km_over_kf
    return matrix


def rotor_site_xml() -> str:
    lines = []
    for i, position in enumerate(ROTOR_POSITIONS, start=1):
        lines.append(
            f'<site name="rotor{i}_site" pos="{position[0]} {position[1]} {position[2]}" '
            'zaxis="0 0 1" size="0.005" type="sphere"/>'
        )
    return "\n      ".join(lines)


def force_model_xml(args: argparse.Namespace) -> str:
    actuators = []
    for i, yaw_sign in enumerate(YAW_SIGNS, start=1):
        actuators.append(
            f"""
    <general name="rotor{i}"
             site="rotor{i}_site"
             gear="0 0 1 0 0 {yaw_sign * args.km_over_kf}"
             dyntype="filterexact"
             dynprm="{args.tau}"
             gaintype="fixed"
             gainprm="1"
             ctrllimited="true"
             ctrlrange="0 {args.force_max}"
             actlimited="true"
             actrange="0 {args.force_max}"
             forcelimited="true"
             forcerange="0 {args.force_max}"
             nsample="{args.nsample}"
             delay="{args.delay}"
             interp="zoh"/>"""
        )

    return f"""
<mujoco model="direct_rotor_force_model">
  <compiler angle="radian"/>
  <option timestep="{args.dt}" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <body name="drone_1" pos="0 0 0">
      <joint name="drone_1" type="free"/>
      <geom type="box" size="0.05 0.05 0.015" mass="1.24"/>
      {rotor_site_xml()}
    </body>
  </worldbody>
  <actuator>
    {"".join(actuators)}
  </actuator>
</mujoco>
"""


def plugin_model_xml(args: argparse.Namespace) -> str:
    actuators = []
    for i, yaw_sign in enumerate(YAW_SIGNS, start=1):
        actuators.append(
            f"""
    <plugin name="rotor{i}"
            site="rotor{i}_site"
            plugin="MujocoRosUtils::RotorMotor"
            actdim="1"
            ctrlrange="0 {args.omega_max}"
            gear="0 0 1 0 0 {yaw_sign * args.km_over_kf}"
            nsample="{args.nsample}"
            delay="{args.delay}"
            interp="zoh">
      <config key="kf" value="{args.kf}"/>
      <config key="tau" value="{args.tau}"/>
    </plugin>"""
        )

    return f"""
<mujoco model="rotor_motor_plugin_model">
  <compiler angle="radian"/>
  <option timestep="{args.dt}" gravity="0 0 0" integrator="RK4"/>
  <extension>
    <plugin plugin="MujocoRosUtils::RotorMotor"/>
  </extension>
  <worldbody>
    <body name="drone_1" pos="0 0 0">
      <joint name="drone_1" type="free"/>
      <geom type="box" size="0.05 0.05 0.015" mass="1.24"/>
      {rotor_site_xml()}
    </body>
  </worldbody>
  <actuator>
    {"".join(actuators)}
  </actuator>
</mujoco>
"""


def load_plugin(plugin_library: Path) -> None:
    global _PLUGIN_LOADED
    if _PLUGIN_LOADED:
        return
    if not plugin_library.exists():
        raise FileNotFoundError(
            f"Plugin library not found: {plugin_library}\n"
            "Build mujoco_ros_utils with MuJoCo 3.10.0 before running this script."
        )
    sibling_core_library = plugin_library.with_name("libMujocoRosUtils.so")
    if sibling_core_library.exists():
        ctypes.CDLL(str(sibling_core_library), mode=ctypes.RTLD_GLOBAL)
    mujoco.mj_loadPluginLibrary(str(plugin_library))
    _PLUGIN_LOADED = True


def make_models(args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjModel, mujoco.MjData]:
    load_plugin(args.plugin_library)
    force_model = mujoco.MjModel.from_xml_string(force_model_xml(args))
    plugin_model = mujoco.MjModel.from_xml_string(plugin_model_xml(args))
    force_data = mujoco.MjData(force_model)
    plugin_data = mujoco.MjData(plugin_model)
    mujoco.mj_forward(force_model, force_data)
    mujoco.mj_forward(plugin_model, plugin_data)
    return force_model, force_data, plugin_model, plugin_data


def commanded_forces(time: float, args: argparse.Namespace) -> np.ndarray:
    phase = 2.0 * np.pi * args.frequency * time + np.asarray(args.phase, dtype=float)
    thrust = args.force_offset + args.force_amplitude * np.sin(phase)
    return np.clip(thrust, 0.0, args.force_max)


def free_joint_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_1")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    return data.qpos[qpos_adr : qpos_adr + 7].copy(), data.qvel[qvel_adr : qvel_adr + 6].copy()


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


def run_comparison(args: argparse.Namespace) -> dict[str, np.ndarray]:
    force_model, force_data, plugin_model, plugin_data = make_models(args)
    matrix = allocation_matrix(args.km_over_kf)
    nsteps = int(np.ceil(args.duration / args.dt))

    initial_force_cmd = commanded_forces(0.0, args)
    initial_omega_cmd = np.sqrt(initial_force_cmd / args.kf)
    initial_omega_cmd = np.clip(initial_omega_cmd, 0.0, args.omega_max)

    for _ in range(int(np.ceil(args.warmup / args.dt))):
        force_data.ctrl[:] = initial_force_cmd
        plugin_data.ctrl[:] = initial_omega_cmd
        mujoco.mj_step(force_model, force_data)
        mujoco.mj_step(plugin_model, plugin_data)

    logs: dict[str, list[np.ndarray] | list[float]] = {
        "time": [],
        "force_cmd": [],
        "omega_cmd": [],
        "force_ctrl": [],
        "force_act": [],
        "force_actuator_force": [],
        "plugin_ctrl": [],
        "plugin_act_omega": [],
        "plugin_actuator_force": [],
        "force_wrench": [],
        "plugin_wrench": [],
        "force_qpos": [],
        "plugin_qpos": [],
        "force_qvel": [],
        "plugin_qvel": [],
    }

    for step in range(nsteps + 1):
        time = step * args.dt
        force_cmd = commanded_forces(time, args)
        omega_cmd = np.sqrt(force_cmd / args.kf)
        omega_cmd = np.clip(omega_cmd, 0.0, args.omega_max)

        force_data.ctrl[:] = force_cmd
        plugin_data.ctrl[:] = omega_cmd
        mujoco.mj_forward(force_model, force_data)
        mujoco.mj_forward(plugin_model, plugin_data)

        force_act = activation_by_actuator(force_model, force_data)
        plugin_act = activation_by_actuator(plugin_model, plugin_data)
        force_actuator_force = force_data.actuator_force.copy()
        plugin_actuator_force = plugin_data.actuator_force.copy()
        force_qpos, force_qvel = free_joint_state(force_model, force_data)
        plugin_qpos, plugin_qvel = free_joint_state(plugin_model, plugin_data)

        logs["time"].append(time)
        logs["force_cmd"].append(force_cmd.copy())
        logs["omega_cmd"].append(omega_cmd.copy())
        logs["force_ctrl"].append(force_data.ctrl.copy())
        logs["force_act"].append(force_act)
        logs["force_actuator_force"].append(force_actuator_force)
        logs["plugin_ctrl"].append(plugin_data.ctrl.copy())
        logs["plugin_act_omega"].append(plugin_act)
        logs["plugin_actuator_force"].append(plugin_actuator_force)
        logs["force_wrench"].append(matrix @ force_actuator_force)
        logs["plugin_wrench"].append(matrix @ plugin_actuator_force)
        logs["force_qpos"].append(force_qpos)
        logs["plugin_qpos"].append(plugin_qpos)
        logs["force_qvel"].append(force_qvel)
        logs["plugin_qvel"].append(plugin_qvel)

        if step < nsteps:
            mujoco.mj_step(force_model, force_data)
            mujoco.mj_step(plugin_model, plugin_data)

    return {key: np.asarray(value) for key, value in logs.items()}


def write_csv(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "force_vs_rotor_motor_plugin.csv"
    header = ["time"]
    for prefix in (
        "force_cmd",
        "omega_cmd",
        "force_ctrl",
        "force_act",
        "force_actuator_force",
        "plugin_ctrl",
        "plugin_act_omega",
        "plugin_actuator_force",
    ):
        header += [f"{prefix}_{i}" for i in range(1, 5)]
    header += [f"force_wrench_{name}" for name in WRENCH_LABELS]
    header += [f"plugin_wrench_{name}" for name in WRENCH_LABELS]
    header += [f"motor_force_error_{i}" for i in range(1, 5)]
    header += [f"wrench_error_{name}" for name in WRENCH_LABELS]
    header += [f"qpos_error_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    header += [f"qvel_error_{name}" for name in ("vx", "vy", "vz", "wx", "wy", "wz")]

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for k, time in enumerate(data["time"]):
            row = [time]
            for prefix in (
                "force_cmd",
                "omega_cmd",
                "force_ctrl",
                "force_act",
                "force_actuator_force",
                "plugin_ctrl",
                "plugin_act_omega",
                "plugin_actuator_force",
            ):
                row += data[prefix][k].tolist()
            row += data["force_wrench"][k].tolist()
            row += data["plugin_wrench"][k].tolist()
            row += (data["plugin_actuator_force"][k] - data["force_actuator_force"][k]).tolist()
            row += (data["plugin_wrench"][k] - data["force_wrench"][k]).tolist()
            row += (data["plugin_qpos"][k] - data["force_qpos"][k]).tolist()
            row += (data["plugin_qvel"][k] - data["force_qvel"][k]).tolist()
            writer.writerow(row)


def plot_motor_forces(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(data["time"], data["force_cmd"][:, i], "k--", linewidth=1.0, label="commanded force")
        axis.plot(data["time"], data["force_actuator_force"][:, i], linewidth=1.0, label="force actuator")
        axis.plot(data["time"], data["plugin_actuator_force"][:, i], linewidth=1.0, label="RotorMotor plugin")
        axis.set_ylabel(f"f{i + 1} [N]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-motor thrust comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "motor_force_comparison.png", dpi=160)
    fig.savefig(output_dir / "motor_force_comparison.pdf")
    plt.close(fig)


def plot_plugin_speed(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(data["time"], data["omega_cmd"][:, i], "k--", linewidth=1.0, label="omega_cmd")
        axis.plot(data["time"], data["plugin_act_omega"][:, i], linewidth=1.0, label="plugin omega")
        axis.set_ylabel(f"omega{i + 1} [rad/s]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("RotorMotor speed state")
    fig.tight_layout()
    fig.savefig(output_dir / "plugin_rotor_speed.png", dpi=160)
    fig.savefig(output_dir / "plugin_rotor_speed.pdf")
    plt.close(fig)


def plot_wrenches(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(data["time"], data["force_wrench"][:, i], linewidth=1.0, label="force actuator")
        axis.plot(data["time"], data["plugin_wrench"][:, i], linewidth=1.0, label="RotorMotor plugin")
        axis.set_ylabel(WRENCH_LABELS[i])
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Aggregate body wrench from actuator_force")
    fig.tight_layout()
    fig.savefig(output_dir / "wrench_comparison.png", dpi=160)
    fig.savefig(output_dir / "wrench_comparison.pdf")
    plt.close(fig)


def plot_errors(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    motor_error = data["plugin_actuator_force"] - data["force_actuator_force"]
    wrench_error = data["plugin_wrench"] - data["force_wrench"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(4):
        axes[0].plot(data["time"], motor_error[:, i], linewidth=1.0, label=f"f{i + 1}")
    for i, label in enumerate(WRENCH_LABELS):
        axes[1].plot(data["time"], wrench_error[:, i], linewidth=1.0, label=label)
    axes[0].set_ylabel("motor force error [N]")
    axes[1].set_ylabel("wrench error")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8, ncol=4)
    fig.suptitle("RotorMotor plugin minus direct force actuator")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_errors.png", dpi=160)
    fig.savefig(output_dir / "comparison_errors.pdf")
    plt.close(fig)


def plot_position(output_dir: Path, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, label in enumerate(("x", "y", "z")):
        axes[i].plot(data["time"], data["force_qpos"][:, i], linewidth=1.0, label="force actuator")
        axes[i].plot(data["time"], data["plugin_qpos"][:, i], linewidth=1.0, label="RotorMotor plugin")
        axes[i].set_ylabel(f"{label} [m]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Free-body position under actuator outputs")
    fig.tight_layout()
    fig.savefig(output_dir / "position_comparison.png", dpi=160)
    fig.savefig(output_dir / "position_comparison.pdf")
    plt.close(fig)


def write_summary(output_dir: Path, args: argparse.Namespace, data: dict[str, np.ndarray]) -> None:
    motor_error = data["plugin_actuator_force"] - data["force_actuator_force"]
    wrench_error = data["plugin_wrench"] - data["force_wrench"]
    qpos_error = data["plugin_qpos"] - data["force_qpos"]
    qvel_error = data["plugin_qvel"] - data["force_qvel"]
    force_max_from_omega = args.kf * args.omega_max * args.omega_max

    lines = [
        "Direct force actuator vs RotorMotor plugin comparison",
        "",
        "Direct force actuator:",
        "  ctrl = desired motor thrust [N]",
        "  act = delayed/filtered motor thrust [N]",
        "  actuator_force = applied motor thrust [N]",
        "",
        "RotorMotor plugin:",
        "  ctrl = desired rotor speed [rad/s]",
        "  act = actual rotor speed [rad/s]",
        "  actuator_force = kf * act^2 [N]",
        "",
        f"kf: {args.kf:.16g} N/(rad/s)^2",
        f"omega_max: {args.omega_max:.6g} rad/s",
        f"kf * omega_max^2: {force_max_from_omega:.6g} N per motor",
        f"force ctrlrange high: {args.force_max:.6g} N per motor",
        f"km_over_kf yaw gear ratio: {args.km_over_kf:.6g}",
        f"tau: {args.tau:.6g} s",
        f"delay: {args.delay:.6g} s",
        f"nsample: {args.nsample}",
        f"dt: {args.dt:.6g} s",
        f"duration: {args.duration:.6g} s",
        f"warmup: {args.warmup:.6g} s",
        "",
        "Commanded motor-force profile:",
        f"  offset: {np.array2string(np.asarray(args.force_offset), precision=6)} N",
        f"  amplitude: {np.array2string(np.asarray(args.force_amplitude), precision=6)} N",
        f"  frequency: {np.array2string(np.asarray(args.frequency), precision=6)} Hz",
        f"  phase: {np.array2string(np.asarray(args.phase), precision=6)} rad",
        "",
        "Comparison metrics, RotorMotor plugin minus direct force actuator:",
        f"  max abs motor thrust error [N]: {np.array2string(np.max(np.abs(motor_error), axis=0), precision=6)}",
        f"  rms motor thrust error [N]: {np.array2string(np.sqrt(np.mean(motor_error**2, axis=0)), precision=6)}",
        f"  max abs wrench error [Fz, Tx, Ty, Tz]: {np.array2string(np.max(np.abs(wrench_error), axis=0), precision=6)}",
        f"  rms wrench error [Fz, Tx, Ty, Tz]: {np.array2string(np.sqrt(np.mean(wrench_error**2, axis=0)), precision=6)}",
        f"  max abs qpos error [x, y, z, qw, qx, qy, qz]: {np.array2string(np.max(np.abs(qpos_error), axis=0), precision=6)}",
        f"  max abs qvel error [vx, vy, vz, wx, wy, wz]: {np.array2string(np.max(np.abs(qvel_error), axis=0), precision=6)}",
        "",
        "Interpretation:",
        "  The steady-state thrust law should match because omega_cmd = sqrt(f_cmd / kf).",
        "  Fast transients do not match exactly because the direct model filters thrust, while",
        "  the plugin filters rotor speed and then squares it to produce thrust.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def save_results(output_dir: Path, args: argparse.Namespace, data: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir, data)
    plot_motor_forces(output_dir, data)
    plot_plugin_speed(output_dir, data)
    plot_wrenches(output_dir, data)
    plot_errors(output_dir, data)
    plot_position(output_dir, data)
    write_summary(output_dir, args, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kf", type=float, default=2.110740925780823e-06)
    parser.add_argument("--omega-max", type=float, default=4000.0)
    parser.add_argument("--km-over-kf", type=float, default=0.015)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--delay", type=float, default=0.018)
    parser.add_argument("--nsample", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.3,
        help="Settle both actuator models at the initial command before logging. Default: 0.3 s.",
    )
    parser.add_argument(
        "--force-max",
        type=float,
        default=None,
        help="Per-motor force limit. Default: kf * omega_max^2.",
    )
    parser.add_argument(
        "--force-offset",
        nargs=4,
        type=float,
        default=(4.6, 4.5, 4.4, 4.7),
        metavar=("F1", "F2", "F3", "F4"),
        help="Per-motor force offset in N.",
    )
    parser.add_argument(
        "--force-amplitude",
        nargs=4,
        type=float,
        default=(1.0, 0.8, 0.7, 0.9),
        metavar=("A1", "A2", "A3", "A4"),
        help="Per-motor force sine amplitude in N.",
    )
    parser.add_argument(
        "--frequency",
        nargs=4,
        type=float,
        default=(0.4, 0.6, 0.8, 0.5),
        metavar=("F1", "F2", "F3", "F4"),
        help="Per-motor force sine frequencies in Hz.",
    )
    parser.add_argument(
        "--phase",
        nargs=4,
        type=float,
        default=(0.0, 1.57079632679, 0.78539816339, 2.35619449019),
        metavar=("P1", "P2", "P3", "P4"),
        help="Per-motor force sine phases in rad.",
    )
    args = parser.parse_args()
    if args.force_max is None:
        args.force_max = args.kf * args.omega_max * args.omega_max
    args.force_offset = np.asarray(args.force_offset, dtype=float)
    args.force_amplitude = np.asarray(args.force_amplitude, dtype=float)
    args.frequency = np.asarray(args.frequency, dtype=float)
    args.phase = np.asarray(args.phase, dtype=float)
    return args


def main() -> None:
    args = parse_args()
    data = run_comparison(args)
    save_results(args.output_dir, args, data)

    motor_error = data["plugin_actuator_force"] - data["force_actuator_force"]
    wrench_error = data["plugin_wrench"] - data["force_wrench"]
    print(f"Saved comparison results to: {args.output_dir}")
    print(f"kf * omega_max^2: {args.kf * args.omega_max * args.omega_max:.6f} N per motor")
    print(
        "Max abs motor thrust error [N]: "
        f"{np.array2string(np.max(np.abs(motor_error), axis=0), precision=6)}"
    )
    print(
        "Max abs wrench error [Fz, Tx, Ty, Tz]: "
        f"{np.array2string(np.max(np.abs(wrench_error), axis=0), precision=6)}"
    )
    print(f"Summary: {args.output_dir / 'summary.txt'}")
    print(
        "Plots: "
        f"{args.output_dir / 'motor_force_comparison.png'}, "
        f"{args.output_dir / 'plugin_rotor_speed.png'}, "
        f"{args.output_dir / 'wrench_comparison.png'}, "
        f"{args.output_dir / 'comparison_errors.png'}, "
        f"{args.output_dir / 'position_comparison.png'}"
    )


if __name__ == "__main__":
    main()
