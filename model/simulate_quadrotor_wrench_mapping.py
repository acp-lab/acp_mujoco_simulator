#!/usr/bin/env python3
"""Run zero-gravity simulations to verify direct vs rotor actuator mapping.

Both models are commanded to produce the same desired body-frame wrench:

    [Fz, Tx, Ty, Tz]

For the direct model, the controls are the direct actuator commands. For the
rotor model, the controls are solved from the allocation matrix read from the
MuJoCo sites/gears at every timestep:

    [Fz, Tx, Ty, Tz] = allocation @ [f1, f2, f3, f4]

The CSV output saves data.ctrl, data.act, data.actuator_force, and the
reconstructed body-frame wrench from data.actuator_force. With the current
<general> actuators, data.act is the delayed/filtered activation state and
data.actuator_force is the final scalar actuator output after gain/force limits.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from compare_quadrotor_actuator_wrenches import (
    BODY_WRENCH_LABELS,
    BODY_WRENCH_ROWS,
    actuator_matrix,
    clip_to_ctrlrange,
    load_actuator_wrenches,
    solve_controls_for_desired_body_wrench,
)


THIS_DIR = Path(__file__).resolve().parent
DIRECT_MODEL = THIS_DIR / "quadrotor_verify_mass_inertia.xml"
ROTOR_MODEL = THIS_DIR / "quadrotor_verify_mass_inertia_delay_and_actuators.xml"
DEFAULT_OUTPUT_DIR = THIS_DIR / "wrench_mapping_results"
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


@dataclass(frozen=True)
class DesiredProfile:
    thrust: float
    torque_offset: np.ndarray
    torque_amplitude: np.ndarray
    torque_frequency: np.ndarray
    torque_phase: np.ndarray
    profile: str


@dataclass(frozen=True)
class SimulationConfig:
    model_path: Path
    output_csv: Path
    body_name: str
    duration: float
    desired_profile: DesiredProfile


def desired_fz_txtytz(profile: DesiredProfile, time: float) -> np.ndarray:
    if profile.profile == "constant":
        torque = profile.torque_offset
    else:
        phase = 2.0 * np.pi * profile.torque_frequency * time + profile.torque_phase
        torque = profile.torque_offset + profile.torque_amplitude * np.sin(phase)
    return np.asarray((profile.thrust, *torque), dtype=float)


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _free_joint_state(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> tuple[np.ndarray, np.ndarray]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, body_name)
    if joint_id < 0:
        return np.full(7, np.nan), np.full(6, np.nan)

    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    return data.qpos[qpos_adr : qpos_adr + 7].copy(), data.qvel[qvel_adr : qvel_adr + 6].copy()


def _csv_header(model: mujoco.MjModel) -> list[str]:
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    header = ["step", "time"]
    header += [f"ctrl_{name}" for name in actuator_names]
    header += _activation_column_names(model)
    header += [f"actuator_force_{name}" for name in actuator_names]
    header += [f"desired_{name}" for name in WRENCH_NAMES]
    header += [f"actual_{name}" for name in WRENCH_NAMES]
    header += [f"error_{name}" for name in WRENCH_NAMES]
    header += ["qpos_x", "qpos_y", "qpos_z", "qpos_qw", "qpos_qx", "qpos_qy", "qpos_qz"]
    header += ["qvel_x", "qvel_y", "qvel_z", "qvel_wx", "qvel_wy", "qvel_wz"]
    return header


def _csv_row(
    step: int,
    time: float,
    ctrl: np.ndarray,
    act: np.ndarray,
    actuator_force: np.ndarray,
    desired_full: np.ndarray,
    actual_full: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> list[float]:
    error = actual_full - desired_full
    return (
        [step, time]
        + ctrl.tolist()
        + act.tolist()
        + actuator_force.tolist()
        + desired_full.tolist()
        + actual_full.tolist()
        + error.tolist()
        + qpos.tolist()
        + qvel.tolist()
    )


def _activation_column_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for actuator_id in range(model.nu):
        actuator_name = model.actuator(actuator_id).name
        act_adr = int(model.actuator_actadr[actuator_id])
        act_num = int(model.actuator_actnum[actuator_id])
        if act_adr < 0 or act_num == 0:
            continue
        if act_num == 1:
            names.append(f"act_{actuator_name}")
        else:
            names += [f"act_{actuator_name}_{i}" for i in range(act_num)]
    return names


def _activation_values_by_actuator(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    values = []
    for actuator_id in range(model.nu):
        act_adr = int(model.actuator_actadr[actuator_id])
        act_num = int(model.actuator_actnum[actuator_id])
        if act_adr < 0 or act_num == 0:
            continue
        values.extend(data.act[act_adr : act_adr + act_num].tolist())
    return np.asarray(values, dtype=float)


def run_simulation(config: SimulationConfig) -> dict[str, np.ndarray | int | float | str]:
    model, wrenches = load_actuator_wrenches(config.model_path, config.body_name)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    matrix6 = actuator_matrix(wrenches)
    nsteps = int(np.ceil(config.duration / model.opt.timestep))
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    times = []
    controls_log = []
    activation_log = []
    actuator_force_log = []
    desired_log = []
    actual_log = []
    qpos_log = []
    qvel_log = []
    allocation_residual_log = []

    with config.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(_csv_header(model))

        for step in range(nsteps + 1):
            time = float(data.time)
            desired_reduced = desired_fz_txtytz(config.desired_profile, time)
            controls, allocation_residual = solve_controls_for_desired_body_wrench(
                matrix6, desired_reduced
            )
            controls = clip_to_ctrlrange(controls, wrenches)

            desired_full = np.zeros(6)
            desired_full[list(BODY_WRENCH_ROWS)] = desired_reduced

            data.ctrl[:] = controls
            mujoco.mj_forward(model, data)

            actuator_force = data.actuator_force.copy()
            activation = _activation_values_by_actuator(model, data)
            actual_full = matrix6 @ actuator_force
            qpos, qvel = _free_joint_state(model, data, config.body_name)
            writer.writerow(
                _csv_row(
                    step,
                    time,
                    data.ctrl.copy(),
                    activation,
                    actuator_force,
                    desired_full,
                    actual_full,
                    qpos,
                    qvel,
                )
            )

            times.append(time)
            controls_log.append(data.ctrl.copy())
            activation_log.append(activation)
            actuator_force_log.append(actuator_force)
            desired_log.append(desired_full)
            actual_log.append(actual_full)
            qpos_log.append(qpos)
            qvel_log.append(qvel)
            allocation_residual_log.append(allocation_residual)

            if step < nsteps:
                mujoco.mj_step(model, data)

    times_array = np.asarray(times)
    controls_array = np.asarray(controls_log)
    activation_array = np.asarray(activation_log)
    actuator_force_array = np.asarray(actuator_force_log)
    desired_array = np.asarray(desired_log)
    actual_array = np.asarray(actual_log)
    errors_array = actual_array - desired_array
    actuator_force_errors_array = actuator_force_array - controls_array
    if activation_array.shape == actuator_force_array.shape:
        actuator_force_activation_errors_array = actuator_force_array - activation_array
        activation_control_errors_array = activation_array - controls_array
    else:
        actuator_force_activation_errors_array = np.zeros((len(times_array), 0))
        activation_control_errors_array = np.zeros((len(times_array), 0))

    return {
        "model": config.model_path.name,
        "csv": str(config.output_csv),
        "nu": model.nu,
        "na": model.na,
        "timestep": float(model.opt.timestep),
        "nsteps": nsteps,
        "times": times_array,
        "controls": controls_array,
        "activation": activation_array,
        "actuator_force": actuator_force_array,
        "desired_wrench": desired_array,
        "actual_wrench": actual_array,
        "qpos": np.asarray(qpos_log),
        "qvel": np.asarray(qvel_log),
        "allocation_residual": np.asarray(allocation_residual_log),
        "control_min": np.min(controls_array, axis=0),
        "control_max": np.max(controls_array, axis=0),
        "activation_min": np.min(activation_array, axis=0) if activation_array.size else np.asarray([]),
        "activation_max": np.max(activation_array, axis=0) if activation_array.size else np.asarray([]),
        "max_abs_wrench_error": np.max(np.abs(errors_array), axis=0),
        "rms_wrench_error": np.sqrt(np.mean(errors_array**2, axis=0)),
        "max_abs_actuator_force_minus_ctrl": np.max(
            np.abs(actuator_force_errors_array), axis=0
        ),
        "max_abs_actuator_force_minus_act": np.max(
            np.abs(actuator_force_activation_errors_array), axis=0
        )
        if actuator_force_activation_errors_array.size
        else np.asarray([]),
        "max_abs_act_minus_ctrl": np.max(
            np.abs(activation_control_errors_array), axis=0
        )
        if activation_control_errors_array.size
        else np.asarray([]),
    }


def write_summary(
    summary_path: Path,
    desired_profile: DesiredProfile,
    summaries: list[dict[str, np.ndarray | int | float | str]],
) -> None:
    lines = [
        "Zero-gravity wrench mapping simulation summary",
        "",
        "With the current XML <general> actuators, model.na is 4.",
        "Use data.ctrl for commanded scalar actuator input, data.act for delayed/filtered activation, and data.actuator_force for final scalar actuator output.",
        "",
        f"desired profile: {desired_profile.profile}",
        f"fixed thrust Fz: {desired_profile.thrust:.6g}",
        f"torque offset [Tx, Ty, Tz]: {_fmt(desired_profile.torque_offset)}",
        f"torque amplitude [Tx, Ty, Tz]: {_fmt(desired_profile.torque_amplitude)}",
        f"torque frequency [Tx, Ty, Tz] Hz: {_fmt(desired_profile.torque_frequency)}",
        f"torque phase [Tx, Ty, Tz] rad: {_fmt(desired_profile.torque_phase)}",
        "",
    ]

    for summary in summaries:
        max_residual = np.max(np.abs(np.asarray(summary["allocation_residual"])), axis=0)
        lines += [
            f"Model: {summary['model']}",
            f"CSV: {summary['csv']}",
            f"nu: {summary['nu']}  na: {summary['na']}  timestep: {summary['timestep']}  nsteps: {summary['nsteps']}",
            f"control min: {_fmt(np.asarray(summary['control_min']))}",
            f"control max: {_fmt(np.asarray(summary['control_max']))}",
            f"activation min: {_fmt(np.asarray(summary['activation_min']))}",
            f"activation max: {_fmt(np.asarray(summary['activation_max']))}",
            f"max abs allocation residual [Fz, Tx, Ty, Tz]: {_fmt(max_residual)}",
            f"max abs wrench error [Fx, Fy, Fz, Tx, Ty, Tz]: {_fmt(np.asarray(summary['max_abs_wrench_error']))}",
            f"rms wrench error [Fx, Fy, Fz, Tx, Ty, Tz]: {_fmt(np.asarray(summary['rms_wrench_error']))}",
            f"max abs act - ctrl: {_fmt(np.asarray(summary['max_abs_act_minus_ctrl']))}",
            f"max abs actuator_force - act: {_fmt(np.asarray(summary['max_abs_actuator_force_minus_act']))}",
            f"max abs actuator_force - ctrl: {_fmt(np.asarray(summary['max_abs_actuator_force_minus_ctrl']))}",
            "",
        ]

    if len(summaries) >= 2:
        direct_actual = np.asarray(summaries[0]["actual_wrench"])
        rotor_actual = np.asarray(summaries[1]["actual_wrench"])
        direct_qpos = np.asarray(summaries[0]["qpos"])
        rotor_qpos = np.asarray(summaries[1]["qpos"])
        lines += [
            "Direct-vs-rotor consistency:",
            f"max abs actual wrench difference [Fx, Fy, Fz, Tx, Ty, Tz]: {_fmt(np.max(np.abs(direct_actual - rotor_actual), axis=0))}",
            f"max abs free-joint qpos difference [x, y, z, qw, qx, qy, qz]: {_fmt(np.max(np.abs(direct_qpos - rotor_qpos), axis=0))}",
            "",
        ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def plot_wrench_tracking(
    summaries: list[dict[str, np.ndarray | int | float | str]],
    output_path: Path,
) -> None:
    desired = np.asarray(summaries[0]["desired_wrench"])
    time = np.asarray(summaries[0]["times"])
    row_indices = [2, 3, 4, 5]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for axis, row, label in zip(axes, row_indices, BODY_WRENCH_LABELS):
        axis.plot(time, desired[:, row], "k--", linewidth=1.2, label=f"desired {label}")
        for summary in summaries:
            actual = np.asarray(summary["actual_wrench"])
            axis.plot(
                np.asarray(summary["times"]),
                actual[:, row],
                linewidth=1.0,
                label=f"{summary['model']} actual",
            )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Desired vs actual body-frame wrench")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_actuator_outputs(
    summaries: list[dict[str, np.ndarray | int | float | str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(summaries), 1, figsize=(10, 6), sharex=False)
    if len(summaries) == 1:
        axes = [axes]

    for axis, summary in zip(axes, summaries):
        time = np.asarray(summary["times"])
        controls = np.asarray(summary["controls"])
        activation = np.asarray(summary["activation"])
        actuator_force = np.asarray(summary["actuator_force"])
        for i in range(controls.shape[1]):
            axis.plot(time, controls[:, i], linewidth=1.0, label=f"ctrl {i}")
            if activation.size:
                axis.plot(
                    time,
                    activation[:, i],
                    "-.",
                    linewidth=0.9,
                    label=f"act {i}",
                )
            axis.plot(
                time,
                actuator_force[:, i],
                "--",
                linewidth=0.9,
                label=f"actuator_force {i}",
            )
        axis.set_title(str(summary["model"]))
        axis.set_ylabel("actuator scalar")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_position_comparison(
    summaries: list[dict[str, np.ndarray | int | float | str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ("x", "y", "z")
    for axis, index, label in zip(axes, range(3), labels):
        for summary in summaries:
            axis.plot(
                np.asarray(summary["times"]),
                np.asarray(summary["qpos"])[:, index],
                linewidth=1.0,
                label=str(summary["model"]),
            )
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Free-body position under matched wrench commands")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate both quadrotor actuator models with gravity removed."
    )
    parser.add_argument("--body", default="drone_1", help="Body frame to use. Default: drone_1.")
    parser.add_argument("--thrust", type=float, default=5.0, help="Desired body Fz. Default: 5.0.")
    parser.add_argument(
        "--profile",
        choices=("sine", "constant"),
        default="sine",
        help="Torque command profile. Default: sine.",
    )
    parser.add_argument(
        "--torque",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=(0.0, 0.0, 0.0),
        help="Torque offset [Tx Ty Tz]. Used as the full torque for --profile constant.",
    )
    parser.add_argument(
        "--torque-amplitude",
        nargs=3,
        type=float,
        metavar=("TX", "TY", "TZ"),
        default=(0.08, 0.05, 0.012),
        help="Sine torque amplitudes [Tx Ty Tz]. Default: 0.08 0.05 0.012.",
    )
    parser.add_argument(
        "--torque-frequency",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        default=(0.7, 1.1, 1.6),
        help="Sine torque frequencies in Hz. Default: 0.7 1.1 1.6.",
    )
    parser.add_argument(
        "--torque-phase",
        nargs=3,
        type=float,
        metavar=("PX", "PY", "PZ"),
        default=(0.0, 1.57079632679, 0.78539816339),
        help="Sine torque phases in radians. Default: 0 pi/2 pi/4.",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV and plot outputs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    profile = DesiredProfile(
        thrust=args.thrust,
        torque_offset=np.asarray(args.torque, dtype=float),
        torque_amplitude=np.asarray(args.torque_amplitude, dtype=float),
        torque_frequency=np.asarray(args.torque_frequency, dtype=float),
        torque_phase=np.asarray(args.torque_phase, dtype=float),
        profile=args.profile,
    )

    configs = [
        SimulationConfig(
            model_path=DIRECT_MODEL.resolve(),
            output_csv=output_dir / "direct_body_wrench_zero_gravity.csv",
            body_name=args.body,
            duration=args.duration,
            desired_profile=profile,
        ),
        SimulationConfig(
            model_path=ROTOR_MODEL.resolve(),
            output_csv=output_dir / "rotor_allocated_wrench_zero_gravity.csv",
            body_name=args.body,
            duration=args.duration,
            desired_profile=profile,
        ),
    ]

    summaries = [run_simulation(config) for config in configs]
    summary_path = output_dir / "summary.txt"
    wrench_plot = output_dir / "wrench_tracking.png"
    actuator_plot = output_dir / "actuator_outputs.png"
    position_plot = output_dir / "position_comparison.png"

    write_summary(summary_path, profile, summaries)
    plot_wrench_tracking(summaries, wrench_plot)
    plot_actuator_outputs(summaries, actuator_plot)
    plot_position_comparison(summaries, position_plot)

    print(f"Torque profile: {profile.profile}")
    print(f"Fixed Fz: {profile.thrust:.6g}")
    print(f"Torque offset [Tx, Ty, Tz]: {_fmt(profile.torque_offset)}")
    print(f"Torque amplitude [Tx, Ty, Tz]: {_fmt(profile.torque_amplitude)}")
    print(f"Torque frequency [Tx, Ty, Tz] Hz: {_fmt(profile.torque_frequency)}")
    for summary in summaries:
        print(f"\n{summary['model']}")
        print(f"  csv: {summary['csv']}")
        print(f"  model.nu={summary['nu']} model.na={summary['na']}")
        print(f"  control min: {_fmt(np.asarray(summary['control_min']))}")
        print(f"  control max: {_fmt(np.asarray(summary['control_max']))}")
        print(f"  activation min: {_fmt(np.asarray(summary['activation_min']))}")
        print(f"  activation max: {_fmt(np.asarray(summary['activation_max']))}")
        print(
            "  max abs wrench error [Fx, Fy, Fz, Tx, Ty, Tz]: "
            f"{_fmt(np.asarray(summary['max_abs_wrench_error']))}"
        )
        print(
            "  max abs actuator_force - act: "
            f"{_fmt(np.asarray(summary['max_abs_actuator_force_minus_act']))}"
        )
        print(
            "  max abs actuator_force - ctrl: "
            f"{_fmt(np.asarray(summary['max_abs_actuator_force_minus_ctrl']))}"
        )
    if len(summaries) >= 2:
        direct_actual = np.asarray(summaries[0]["actual_wrench"])
        rotor_actual = np.asarray(summaries[1]["actual_wrench"])
        print(
            "\nmax abs direct actual - rotor actual [Fx, Fy, Fz, Tx, Ty, Tz]: "
            f"{_fmt(np.max(np.abs(direct_actual - rotor_actual), axis=0))}"
        )
    print(f"\nsummary: {summary_path}")
    print(f"plots: {wrench_plot}, {actuator_plot}, {position_plot}")


if __name__ == "__main__":
    main()
