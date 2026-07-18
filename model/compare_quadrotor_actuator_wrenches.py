#!/usr/bin/env python3
"""Compare direct body-wrench actuators with per-rotor force actuators.

For a MuJoCo site motor with a 6D gear vector, this script reports the
body-frame wrench applied to the actuator body:

    [Fx, Fy, Fz, Tx, Ty, Tz]

The torque is taken about the MuJoCo body frame origin:

    tau_body = r_site_body x force_body + gear_torque_body

This makes the rotor-force model directly comparable with the model that uses
one thrust actuator and three body-moment actuators.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = (
    THIS_DIR / "quadrotor_verify_mass_inertia.xml",
    THIS_DIR / "quadrotor_verify_mass_inertia_delay_and_actuators.xml",
)
BODY_WRENCH_ROWS = (2, 3, 4, 5)
BODY_WRENCH_LABELS = ("Fz", "Tx", "Ty", "Tz")


@dataclass(frozen=True)
class ActuatorWrench:
    name: str
    site_name: str
    site_position_body: np.ndarray
    gear: np.ndarray
    unit_wrench_body: np.ndarray
    ctrlrange: tuple[float, float] | None


def _is_site_actuator(model: mujoco.MjModel, actuator_id: int) -> bool:
    return int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_SITE)


def _rotation(matrix9: np.ndarray) -> np.ndarray:
    return np.asarray(matrix9, dtype=float).reshape(3, 3)


def _body_frame_wrench_for_unit_actuator(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: int,
    body_id: int,
) -> ActuatorWrench:
    if not _is_site_actuator(model, actuator_id):
        raise ValueError(
            f"{model.actuator(actuator_id).name!r} is not a site actuator; "
            "this checker only handles site motors."
        )

    site_id = int(model.actuator_trnid[actuator_id, 0])
    site_body_id = int(model.site_bodyid[site_id])
    if site_body_id != body_id:
        raise ValueError(
            f"{model.actuator(actuator_id).name!r} is attached to body "
            f"{model.body(site_body_id).name!r}, not {model.body(body_id).name!r}."
        )

    body_world_from_body = _rotation(data.xmat[body_id])
    site_world_from_site = _rotation(data.site_xmat[site_id])
    body_from_world = body_world_from_body.T
    body_from_site = body_from_world @ site_world_from_site

    gear = np.asarray(model.actuator_gear[actuator_id, :6], dtype=float)
    force_body = body_from_site @ gear[:3]
    torque_body_at_site = body_from_site @ gear[3:6]
    site_position_body = body_from_world @ (data.site_xpos[site_id] - data.xpos[body_id])
    torque_body = np.cross(site_position_body, force_body) + torque_body_at_site
    unit_wrench_body = np.concatenate((force_body, torque_body))

    ctrlrange = None
    if int(model.actuator_ctrllimited[actuator_id]):
        lo, hi = np.asarray(model.actuator_ctrlrange[actuator_id], dtype=float)
        ctrlrange = (float(lo), float(hi))

    return ActuatorWrench(
        name=model.actuator(actuator_id).name,
        site_name=model.site(site_id).name,
        site_position_body=site_position_body,
        gear=gear,
        unit_wrench_body=unit_wrench_body,
        ctrlrange=ctrlrange,
    )


def load_actuator_wrenches(model_path: Path, body_name: str) -> tuple[mujoco.MjModel, list[ActuatorWrench]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body {body_name!r} does not exist in {model_path}.")

    wrenches: list[ActuatorWrench] = []
    for actuator_id in range(model.nu):
        wrenches.append(
            _body_frame_wrench_for_unit_actuator(model, data, actuator_id, body_id)
        )
    return model, wrenches


def actuator_matrix(wrenches: Iterable[ActuatorWrench]) -> np.ndarray:
    return np.column_stack([w.unit_wrench_body for w in wrenches])


def clip_to_ctrlrange(ctrl: np.ndarray, wrenches: list[ActuatorWrench]) -> np.ndarray:
    clipped = np.asarray(ctrl, dtype=float).copy()
    for i, wrench in enumerate(wrenches):
        if wrench.ctrlrange is not None:
            lo, hi = wrench.ctrlrange
            clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def solve_controls_for_desired_body_wrench(
    matrix6: np.ndarray,
    desired_fz_txtytz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reduced = matrix6[list(BODY_WRENCH_ROWS), :]
    controls, *_ = np.linalg.lstsq(reduced, desired_fz_txtytz, rcond=None)
    residual = reduced @ controls - desired_fz_txtytz
    return controls, residual


def _fmt_vector(values: np.ndarray, precision: int = 6) -> str:
    return "[" + ", ".join(f"{value: .{precision}g}" for value in values) + "]"


def print_model_report(
    model_path: Path,
    body_name: str,
    user_ctrl: np.ndarray | None,
    desired_body_wrench: np.ndarray | None,
) -> None:
    model, wrenches = load_actuator_wrenches(model_path, body_name)
    matrix = actuator_matrix(wrenches)

    print(f"\n=== {model_path.name} ===")
    print(f"body: {body_name}")
    print(f"actuators: {[w.name for w in wrenches]}")
    print("\nPer-unit actuator body-frame wrench [Fx, Fy, Fz, Tx, Ty, Tz]:")
    for wrench in wrenches:
        ctrlrange = "" if wrench.ctrlrange is None else f" ctrlrange={wrench.ctrlrange}"
        print(
            f"  {wrench.name:24s} site={wrench.site_name:22s} "
            f"r_body={_fmt_vector(wrench.site_position_body)} "
            f"wrench={_fmt_vector(wrench.unit_wrench_body)}{ctrlrange}"
        )

    print("\nAllocation matrix rows [Fx, Fy, Fz, Tx, Ty, Tz], columns are actuators:")
    print(np.array2string(matrix, precision=6, suppress_small=True))

    if user_ctrl is not None:
        if user_ctrl.size != model.nu:
            raise ValueError(
                f"{model_path.name} has {model.nu} actuators, but --ctrl has "
                f"{user_ctrl.size} values."
            )
        ctrl = clip_to_ctrlrange(user_ctrl, wrenches)
        total = matrix @ ctrl
        print("\nTotal body-frame wrench from --ctrl:")
        print(f"  requested ctrl: {_fmt_vector(user_ctrl)}")
        if not np.allclose(ctrl, user_ctrl):
            print(f"  clipped ctrl:   {_fmt_vector(ctrl)}")
        print(f"  wrench:         {_fmt_vector(total)}")

    if desired_body_wrench is not None:
        controls, residual = solve_controls_for_desired_body_wrench(
            matrix, desired_body_wrench
        )
        clipped = clip_to_ctrlrange(controls, wrenches)
        achieved = matrix[list(BODY_WRENCH_ROWS), :] @ clipped
        full_achieved = matrix @ clipped
        print("\nControls that match desired [Fz, Tx, Ty, Tz]:")
        print(f"  desired {BODY_WRENCH_LABELS}: {_fmt_vector(desired_body_wrench)}")
        print(f"  solved ctrl:             {_fmt_vector(controls)}")
        if not np.allclose(clipped, controls):
            print(f"  clipped ctrl:            {_fmt_vector(clipped)}")
        print(f"  residual before clipping:{_fmt_vector(residual)}")
        print(f"  achieved {BODY_WRENCH_LABELS}: {_fmt_vector(achieved)}")
        print(f"  achieved full wrench:    {_fmt_vector(full_achieved)}")

        for i, wrench in enumerate(wrenches):
            if wrench.ctrlrange is None:
                continue
            lo, hi = wrench.ctrlrange
            if controls[i] < lo or controls[i] > hi:
                print(
                    f"  WARNING: {wrench.name} requires {controls[i]:.6g}, "
                    f"outside ctrlrange {wrench.ctrlrange}."
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one or both quadrotor MuJoCo XML files and report the "
            "body-frame wrench produced by each actuator."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        help=(
            "XML model to inspect. Can be passed more than once. "
            "Default: both verification quadrotor XML files."
        ),
    )
    parser.add_argument(
        "--body",
        default="drone_1",
        help="MuJoCo body name whose body frame should be used. Default: drone_1.",
    )
    parser.add_argument(
        "--ctrl",
        nargs="+",
        type=float,
        help=(
            "Optional actuator controls for each loaded model. The script prints "
            "the resulting total body-frame wrench."
        ),
    )
    parser.add_argument(
        "--desired-wrench",
        nargs=4,
        type=float,
        metavar=("FZ", "TX", "TY", "TZ"),
        default=(5.0, 0.0, 0.0, 0.0),
        help=(
            "Desired direct body wrench [Fz Tx Ty Tz]. The script solves the "
            "actuator controls that reproduce it. Default: 5.0 0.05 -0.02 0.01."
        ),
    )
    parser.add_argument(
        "--no-desired-wrench",
        action="store_true",
        help="Do not solve actuator controls for a desired [Fz Tx Ty Tz] wrench.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = args.model if args.model else list(DEFAULT_MODELS)
    user_ctrl = None if args.ctrl is None else np.asarray(args.ctrl, dtype=float)
    desired = (
        None
        if args.no_desired_wrench
        else np.asarray(args.desired_wrench, dtype=float)
    )

    for model_path in model_paths:
        print_model_report(model_path.resolve(), args.body, user_ctrl, desired)


if __name__ == "__main__":
    main()
