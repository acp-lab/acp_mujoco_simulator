#!/usr/bin/env python3
"""Validate the quadrotor acro rotor allocator from quadrotor_acro_macro.xml.xacro.

The script reads the rotor site positions, actuator gears, kf, and omega limits
from the xacro file, builds the same reduced wrench matrix used by AcroMode:

    [Fz, Mx, My, Mz] = A @ [f1, f2, f3, f4]

where each column is:

    force = gear[0:3]
    moment = r x force + gear[3:6]

It then sends a time-varying desired wrench, solves bounded motor forces with
OSQP when available, converts forces to motor speeds, and saves plots/results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_XACRO = THIS_DIR / "quadrotor_acro_macro.xml.xacro"
DEFAULT_OUTPUT_DIR = THIS_DIR / "quadrotor_acro_allocator_validation_results"
WRENCH_NAMES = ("Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class RotorConfig:
    name: str
    site: str
    position: np.ndarray
    gear: np.ndarray
    kf: float
    omega_min: float
    omega_max: float


class ScipyBoundedAllocator:
    name = "scipy_lsq_linear"

    def __init__(
        self,
        allocation: np.ndarray,
        force_min: np.ndarray,
        force_max: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.allocation = allocation
        self.force_min = force_min
        self.force_max = force_max
        self.weights = weights
        self.weighted_allocation = weights[:, None] * allocation

    def solve(self, desired: np.ndarray) -> tuple[np.ndarray, str]:
        from scipy.optimize import lsq_linear

        result = lsq_linear(
            self.weighted_allocation,
            self.weights * desired,
            bounds=(self.force_min, self.force_max),
            lsmr_tol="auto",
        )
        return result.x, f"{self.name}_status_{result.status}"


class OsqpBoundedAllocator:
    name = "osqp"

    def __init__(
        self,
        allocation: np.ndarray,
        force_min: np.ndarray,
        force_max: np.ndarray,
        weights: np.ndarray,
        regularization: float,
    ) -> None:
        import osqp
        import scipy.sparse as sp

        self.allocation = allocation
        self.force_min = force_min
        self.force_max = force_max
        self.weights = weights
        self.weighted_allocation = weights[:, None] * allocation
        nu = allocation.shape[1]

        p_dense = 2.0 * self.weighted_allocation.T @ self.weighted_allocation
        p_dense += regularization * np.eye(nu)
        p_dense = 0.5 * (p_dense + p_dense.T)

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sp.csc_matrix(np.triu(p_dense)),
            q=np.zeros(nu),
            A=sp.eye(nu, format="csc"),
            l=force_min,
            u=force_max,
            verbose=False,
            warm_starting=True,
            polish=False,
            eps_abs=1.0e-7,
            eps_rel=1.0e-7,
            max_iter=4000,
        )

    def solve(self, desired: np.ndarray) -> tuple[np.ndarray, str]:
        q = -2.0 * self.weighted_allocation.T @ (self.weights * desired)
        self.solver.update(q=q)
        result = self.solver.solve()
        status = getattr(result.info, "status", "unknown")
        if result.x is None or "solved" not in status.lower():
            raise RuntimeError(f"OSQP failed with status {status}")
        return np.clip(result.x, self.force_min, self.force_max), f"osqp_{status}"


def parse_vector(value: str, expected_size: int) -> np.ndarray:
    values = np.fromstring(value, sep=" ", dtype=float)
    if values.shape != (expected_size,):
        raise ValueError(
            f"Expected {expected_size} values in {value!r}, got {values.shape[0]}"
        )
    return values


def strip_xacro_namespace(xml_text: str) -> str:
    return re.sub(r"\s+xmlns:xacro=\"[^\"]+\"", "", xml_text).replace("xacro:", "")


def load_rotor_configs(xacro_path: Path) -> list[RotorConfig]:
    root = ET.fromstring(strip_xacro_namespace(xacro_path.read_text()))

    sites: dict[str, np.ndarray] = {}
    for site in root.findall(".//site"):
        name = site.get("name")
        pos = site.get("pos")
        if name and pos and "_rotor" in name and name.endswith("_site"):
            sites[name] = parse_vector(pos, 3)

    rotors: list[RotorConfig] = []
    for actuator in root.findall(".//plugin"):
        if actuator.get("plugin") != "MujocoRosUtils::RotorMotor":
            continue
        name = actuator.get("name")
        site = actuator.get("site")
        gear = actuator.get("gear")
        ctrlrange = actuator.get("ctrlrange")
        if not name or not site or not gear or not ctrlrange:
            raise ValueError(
                f"Incomplete rotor actuator definition: {ET.tostring(actuator, encoding='unicode')}"
            )

        kf = None
        for config in actuator.findall("./config"):
            if config.get("key") == "kf":
                kf = float(config.get("value"))
                break
        if kf is None:
            raise ValueError(f"Missing kf config for {name}")

        if site not in sites:
            raise ValueError(f"Rotor actuator {name} references unknown site {site}")

        omega_min, omega_max = parse_vector(ctrlrange, 2)
        rotors.append(
            RotorConfig(
                name=name,
                site=site,
                position=sites[site],
                gear=parse_vector(gear, 6),
                kf=kf,
                omega_min=float(omega_min),
                omega_max=float(omega_max),
            )
        )

    rotors.sort(key=lambda item: item.name)
    if len(rotors) != 4:
        raise ValueError(f"Expected 4 rotor actuators, found {len(rotors)}")
    return rotors


def allocation_matrix(rotors: list[RotorConfig]) -> np.ndarray:
    columns = []
    for rotor in rotors:
        force = rotor.gear[:3]
        gear_torque = rotor.gear[3:]
        torque = np.cross(rotor.position, force) + gear_torque
        columns.append(
            np.array([force[2], torque[0], torque[1], torque[2]], dtype=float)
        )
    return np.column_stack(columns)


def make_allocator(
    allocation: np.ndarray,
    force_min: np.ndarray,
    force_max: np.ndarray,
    weights: np.ndarray,
    regularization: float,
    backend: str,
):
    if backend == "osqp":
        try:
            return OsqpBoundedAllocator(
                allocation, force_min, force_max, weights, regularization
            )
        except Exception as exc:
            print(f"OSQP unavailable ({exc}); using scipy bounded least squares.")
    elif backend != "scipy":
        raise ValueError(f"Unsupported allocator backend: {backend}")
    return ScipyBoundedAllocator(allocation, force_min, force_max, weights)


def desired_wrench_profile(time: np.ndarray) -> np.ndarray:
    desired = np.empty((time.size, 4), dtype=float)
    desired[:, 0] = 12.0 + 2.0 * np.sin(0.7 * time) + 0.8 * np.sin(2.2 * time)
    desired[:, 1] = 0.14 * np.sin(1.3 * time)
    desired[:, 2] = 0.12 * np.cos(1.1 * time)
    desired[:, 3] = 0.018 * np.sin(0.9 * time)
    return desired


def save_plots(
    output_dir: Path,
    time: np.ndarray,
    desired: np.ndarray,
    achieved: np.ndarray,
    forces: np.ndarray,
    omegas: np.ndarray,
    residual: np.ndarray,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(
            time, desired[:, i], label=f"desired {WRENCH_NAMES[i]}", linewidth=2.0
        )
        axis.plot(
            time,
            achieved[:, i],
            "--",
            label=f"allocated {WRENCH_NAMES[i]}",
            linewidth=1.5,
        )
        axis.grid(True)
        axis.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "desired_vs_allocated_wrench.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i in range(forces.shape[1]):
        axes[0].plot(time, forces[:, i], label=f"f{i + 1}")
        axes[1].plot(time, omegas[:, i], label=f"omega{i + 1}")
    axes[0].set_ylabel("motor force [N]")
    axes[1].set_ylabel("motor speed [rad/s]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True)
        axis.legend(loc="best", ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "motor_forces_and_speeds.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, axis in enumerate(axes):
        axis.plot(time, residual[:, i], label=f"{WRENCH_NAMES[i]} residual")
        axis.grid(True)
        axis.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "allocation_residual.png", dpi=180)
    plt.close(fig)


def write_csv(
    path: Path,
    time: np.ndarray,
    desired: np.ndarray,
    achieved: np.ndarray,
    forces: np.ndarray,
    omegas: np.ndarray,
    residual: np.ndarray,
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["time"]
        header += [f"desired_{name}" for name in WRENCH_NAMES]
        header += [f"allocated_{name}" for name in WRENCH_NAMES]
        header += [f"residual_{name}" for name in WRENCH_NAMES]
        header += [f"force_motor_{i + 1}" for i in range(4)]
        header += [f"omega_motor_{i + 1}" for i in range(4)]
        writer.writerow(header)
        for row in range(time.size):
            writer.writerow(
                [time[row]]
                + desired[row].tolist()
                + achieved[row].tolist()
                + residual[row].tolist()
                + forces[row].tolist()
                + omegas[row].tolist()
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xacro", type=Path, default=DEFAULT_XACRO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--allocator", choices=("osqp", "scipy"), default="osqp")
    parser.add_argument("--regularization", type=float, default=1.0e-8)
    parser.add_argument(
        "--weights", nargs=4, type=float, default=(1.0, 20.0, 20.0, 5.0)
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rotors = load_rotor_configs(args.xacro)
    allocation = allocation_matrix(rotors)
    kf = np.array([rotor.kf for rotor in rotors], dtype=float)
    omega_min = np.array([rotor.omega_min for rotor in rotors], dtype=float)
    omega_max = np.array([rotor.omega_max for rotor in rotors], dtype=float)
    force_min = kf * omega_min * omega_min
    force_max = kf * omega_max * omega_max
    weights = np.array(args.weights, dtype=float)

    allocator = make_allocator(
        allocation, force_min, force_max, weights, args.regularization, args.allocator
    )

    time = np.arange(0.0, args.duration + 0.5 * args.dt, args.dt)
    desired = desired_wrench_profile(time)
    forces = np.empty((time.size, 4), dtype=float)
    achieved = np.empty((time.size, 4), dtype=float)
    residual = np.empty((time.size, 4), dtype=float)
    status_counts: dict[str, int] = {}

    for i, wrench in enumerate(desired):
        motor_force, status = allocator.solve(wrench)
        forces[i] = motor_force
        achieved[i] = allocation @ motor_force
        residual[i] = achieved[i] - wrench
        status_counts[status] = status_counts.get(status, 0) + 1

    omegas = np.sqrt(np.maximum(forces, 0.0) / kf[None, :])
    omega_squared_allocation = allocation @ np.diag(kf)

    write_csv(
        output_dir / "quadrotor_acro_allocator_validation.csv",
        time,
        desired,
        achieved,
        forces,
        omegas,
        residual,
    )
    save_plots(output_dir, time, desired, achieved, forces, omegas, residual)

    summary = {
        "xacro": str(args.xacro),
        "allocator": allocator.name,
        "status_counts": status_counts,
        "weights": weights.tolist(),
        "regularization": args.regularization,
        "rotors": [
            {
                "name": rotor.name,
                "site": rotor.site,
                "position": rotor.position.tolist(),
                "gear": rotor.gear.tolist(),
                "kf": rotor.kf,
                "omega_min": rotor.omega_min,
                "omega_max": rotor.omega_max,
                "force_min": float(force_min[i]),
                "force_max": float(force_max[i]),
            }
            for i, rotor in enumerate(rotors)
        ],
        "allocation_force_to_wrench": allocation.tolist(),
        "allocation_omega_squared_to_wrench": omega_squared_allocation.tolist(),
        "rank": int(np.linalg.matrix_rank(allocation)),
        "condition_number": float(np.linalg.cond(allocation)),
        "singular_values": np.linalg.svd(allocation, compute_uv=False).tolist(),
        "max_abs_residual": np.max(np.abs(residual), axis=0).tolist(),
        "rms_residual": np.sqrt(np.mean(residual * residual, axis=0)).tolist(),
        "force_min_observed": np.min(forces, axis=0).tolist(),
        "force_max_observed": np.max(forces, axis=0).tolist(),
        "omega_min_observed": np.min(omegas, axis=0).tolist(),
        "omega_max_observed": np.max(omegas, axis=0).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savetxt(output_dir / "allocation_matrix_force_to_wrench.txt", allocation)
    np.savetxt(
        output_dir / "allocation_matrix_omega_squared_to_wrench.txt",
        omega_squared_allocation,
    )

    print(f"Loaded: {args.xacro}")
    print(f"Saved results in: {output_dir}")
    print(f"Allocator: {allocator.name}")
    print(f"Statuses: {status_counts}")
    print(f"Allocation matrix [Fz, Mx, My, Mz] = A @ [f1, f2, f3, f4]:\n{allocation}")
    print(f"Rank: {summary['rank']}")
    print(f"Condition number: {summary['condition_number']:.6g}")
    print(f"Per-motor force max [N]: {force_max}")
    print(
        f"Observed motor force range [N]: min {np.min(forces, axis=0)}, max {np.max(forces, axis=0)}"
    )
    print(f"Observed motor omega max [rad/s]: {np.max(omegas, axis=0)}")
    print(f"Max abs residual [Fz, Mx, My, Mz]: {np.max(np.abs(residual), axis=0)}")


if __name__ == "__main__":
    main()
