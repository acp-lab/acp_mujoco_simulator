#!/usr/bin/env python3
"""Validate the MujocoRosUtils::RotorMotor actuator plugin.

The test model contains one site actuator whose control is desired rotor speed
omega_cmd [rad/s]. The plugin stores actual rotor speed omega in mjData.act and
outputs scalar actuator force

    thrust = kf * omega^2.

The actuator gear maps that scalar thrust to force and yaw torque at the site.
"""

from __future__ import annotations

import argparse
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
DEFAULT_PLUGIN = (
    WS_DIR / "mujoco-3.10.0" / "bin" / "mujoco_plugin" / "libMujocoRosUtilsPlugin.so"
)
_PLUGIN_LOADED = False


def make_model_xml(
    kf: float, tau: float, km_over_kf: float, omega_max: float, dt: float
) -> str:
    return f"""
<mujoco model="rotor_motor_plugin_validation">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 0" integrator="Euler"/>
  <extension>
    <plugin plugin="MujocoRosUtils::RotorMotor"/>
  </extension>
  <worldbody>
    <body name="body" pos="0 0 0">
      <joint type="free"/>
      <geom type="box" size="0.03 0.03 0.01" mass="1.0"/>
      <site name="rotor_site" pos="0.1 0.1 0" zaxis="0 0 1" size="0.005" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <plugin name="rotor"
            site="rotor_site"
            plugin="MujocoRosUtils::RotorMotor"
            actdim="1"
            ctrlrange="0 {omega_max}"
            gear="0 0 1 0 0 {km_over_kf}">
      <config key="kf" value="{kf}"/>
      <config key="tau" value="{tau}"/>
    </plugin>
  </actuator>
</mujoco>
"""


def load_model(args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
    global _PLUGIN_LOADED
    if not args.plugin_library.exists():
        raise FileNotFoundError(
            f"Plugin library not found: {args.plugin_library}\n"
            "Build mujoco_ros_utils and source install/setup.bash first."
        )
    if not _PLUGIN_LOADED:
        sibling_core_library = args.plugin_library.with_name("libMujocoRosUtils.so")
        if sibling_core_library.exists():
            ctypes.CDLL(str(sibling_core_library), mode=ctypes.RTLD_GLOBAL)
        mujoco.mj_loadPluginLibrary(str(args.plugin_library))
        _PLUGIN_LOADED = True
    model = mujoco.MjModel.from_xml_string(
        make_model_xml(args.kf, args.tau, args.km_over_kf, args.omega_max, args.dt)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def validate_static_force_law(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    kf: float,
    omega_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actuator_id = 0
    actadr = model.actuator_actadr[actuator_id]
    measured = []
    expected = []
    for omega in omega_values:
        data.act[actadr] = omega
        mujoco.mj_forward(model, data)
        measured.append(float(data.actuator_force[actuator_id]))
        expected.append(kf * omega * omega)
    return np.asarray(measured), np.asarray(expected)


def simulate_step_response(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    actuator_id = 0
    actadr = model.actuator_actadr[actuator_id]
    times = np.arange(0.0, args.duration + args.dt, args.dt)
    omega = np.zeros_like(times)
    omega_expected = np.zeros_like(times)
    thrust = np.zeros_like(times)
    thrust_expected = np.zeros_like(times)
    act_dot = np.zeros_like(times)
    ctrl = np.zeros_like(times)

    data.act[actadr] = 0.0
    data.ctrl[actuator_id] = 0.0
    mujoco.mj_forward(model, data)

    for k, t in enumerate(times):
        data.ctrl[actuator_id] = args.omega_cmd
        mujoco.mj_forward(model, data)
        ctrl[k] = data.ctrl[actuator_id]
        omega[k] = data.act[actadr]
        act_dot[k] = data.act_dot[actadr]
        thrust[k] = data.actuator_force[actuator_id]
        omega_expected[k] = args.omega_cmd * (1.0 - np.exp(-t / args.tau))
        thrust_expected[k] = args.kf * omega_expected[k] * omega_expected[k]
        mujoco.mj_step(model, data)

    return {
        "time": times,
        "ctrl_omega_cmd": ctrl,
        "omega": omega,
        "omega_expected": omega_expected,
        "act_dot": act_dot,
        "thrust": thrust,
        "thrust_expected": thrust_expected,
    }


def save_results(
    output_dir: Path,
    static_omega: np.ndarray,
    static_measured: np.ndarray,
    static_expected: np.ndarray,
    step: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    static_data = np.column_stack(
        (
            static_omega,
            static_measured,
            static_expected,
            static_measured - static_expected,
        )
    )
    np.savetxt(
        output_dir / "static_force_law.csv",
        static_data,
        delimiter=",",
        header="omega_rad_s,measured_thrust_N,expected_thrust_N,error_N",
        comments="",
    )

    step_data = np.column_stack(
        (
            step["time"],
            step["ctrl_omega_cmd"],
            step["omega"],
            step["omega_expected"],
            step["act_dot"],
            step["thrust"],
            step["thrust_expected"],
        )
    )
    np.savetxt(
        output_dir / "step_response.csv",
        step_data,
        delimiter=",",
        header="time_s,omega_cmd_rad_s,omega_rad_s,omega_expected_rad_s,omega_dot_rad_s2,thrust_N,thrust_expected_N",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(static_omega, static_expected, "k--", label="kf omega^2")
    ax.plot(static_omega, static_measured, "o", label="plugin")
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("thrust [N]")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "static_force_law.png", dpi=160)
    fig.savefig(output_dir / "static_force_law.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(step["time"], step["ctrl_omega_cmd"], "k--", label="omega_cmd")
    axes[0].plot(
        step["time"],
        step["omega_expected"],
        "tab:gray",
        linestyle=":",
        label="first-order expected",
    )
    axes[0].plot(step["time"], step["omega"], label="plugin omega")
    axes[0].set_ylabel("omega [rad/s]")
    axes[0].grid(True)
    axes[0].legend(loc="best")
    axes[1].plot(
        step["time"],
        step["thrust_expected"],
        "tab:gray",
        linestyle=":",
        label="kf omega_expected^2",
    )
    axes[1].plot(step["time"], step["thrust"], label="plugin thrust")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("thrust [N]")
    axes[1].grid(True)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "step_response.png", dpi=160)
    fig.savefig(output_dir / "step_response.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-library", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument(
        "--output-dir", type=Path, default=THIS_DIR / "rotor_motor_validation_results"
    )
    parser.add_argument("--kf", type=float, default=2.110740925780823e-06)
    parser.add_argument("--km-over-kf", type=float, default=0.015)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--omega-max", type=float, default=4000.0)
    parser.add_argument("--omega-cmd", type=float, default=2000.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_model(args)
    static_omega = np.linspace(0.0, args.omega_max, 11)
    static_measured, static_expected = validate_static_force_law(
        model, data, args.kf, static_omega
    )
    static_error = np.max(np.abs(static_measured - static_expected))

    model, data = load_model(args)
    step = simulate_step_response(model, data, args)
    step_error = np.max(np.abs(step["thrust"] - step["thrust_expected"]))

    save_results(args.output_dir, static_omega, static_measured, static_expected, step)

    print(f"Saved validation results to: {args.output_dir}")
    print(f"Static force law max abs error: {static_error:.6e} N")
    print(
        f"Step response max abs thrust difference from continuous first-order reference: {step_error:.6e} N"
    )
    print(f"Final omega: {step['omega'][-1]:.6f} rad/s")
    print(f"Final thrust: {step['thrust'][-1]:.6f} N")


if __name__ == "__main__":
    main()
