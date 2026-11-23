#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from robo_manip_baselines.common import DataKey, RmbData

JOINT_NAMES = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
    "left_j7",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
    "right_j7",
    "right_gripper",
]

DYNAMIXEL_NAMES = [
    "left_p_id33",
    "left_p_id32",
    "left_x_id16",
    "left_p_id31",
    "left_x_id15",
    "left_x_id13",
    "left_x_id14",
    "left_x_id12",
    "left_x_id11",
    "right_p_id43",
    "right_p_id42",
    "right_x_id26",
    "right_p_id41",
    "right_x_id25",
    "right_x_id23",
    "right_x_id24",
    "right_x_id22",
    "right_x_id21",
]

COMMAND_DYNAMIXEL_RAW_GOAL = "command_dynamixel_raw_goal"
COMMAND_DYNAMIXEL_RAW_GOAL_UNCLIPPED = "command_dynamixel_raw_goal_unclipped"
MEASURED_DYNAMIXEL_RAW_POS = "measured_dynamixel_raw_pos"
DYNAMIXEL_POSITION_LIMIT_MIN = "dynamixel_position_limit_min"
DYNAMIXEL_POSITION_LIMIT_MAX = "dynamixel_position_limit_max"
MEASURED_DYNAMIXEL_PRESENT_CURRENT_RAW = "measured_dynamixel_present_current_raw"
MEASURED_DYNAMIXEL_PRESENT_PWM_RAW = "measured_dynamixel_present_pwm_raw"
DYNAMIXEL_CURRENT_LIMIT_RAW = "dynamixel_current_limit_raw"
DYNAMIXEL_PWM_LIMIT_RAW = "dynamixel_pwm_limit_raw"


def load_data(rmb_path):
    with RmbData(rmb_path) as rmb_data:
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        command = np.asarray(rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64)
        measured = np.asarray(rmb_data[DataKey.MEASURED_JOINT_POS][:], dtype=np.float64)

    if command.shape != measured.shape:
        raise ValueError(
            f"command and measured shapes differ: {command.shape} vs {measured.shape}."
        )
    if command.ndim != 2 or command.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"Expected joint data shape to be (N, {len(JOINT_NAMES)}), "
            f"but got {command.shape}."
        )
    if time.shape[0] != command.shape[0]:
        raise ValueError(
            f"time and joint sample counts differ: {time.shape[0]} vs {command.shape[0]}."
        )

    return time, command, measured


def load_dynamixel_raw_data(rmb_path):
    required_keys = [
        COMMAND_DYNAMIXEL_RAW_GOAL,
        COMMAND_DYNAMIXEL_RAW_GOAL_UNCLIPPED,
        MEASURED_DYNAMIXEL_RAW_POS,
        DYNAMIXEL_POSITION_LIMIT_MIN,
        DYNAMIXEL_POSITION_LIMIT_MAX,
    ]
    with RmbData(rmb_path) as rmb_data:
        missing_keys = [key for key in required_keys if key not in rmb_data]
        if missing_keys:
            raise ValueError(
                "RMB file does not contain Dynamixel raw debug keys: "
                + ", ".join(missing_keys)
            )
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        command = np.asarray(rmb_data[COMMAND_DYNAMIXEL_RAW_GOAL][:], dtype=np.float64)
        command_unclipped = np.asarray(
            rmb_data[COMMAND_DYNAMIXEL_RAW_GOAL_UNCLIPPED][:], dtype=np.float64
        )
        measured = np.asarray(rmb_data[MEASURED_DYNAMIXEL_RAW_POS][:], dtype=np.float64)
        limit_min = np.asarray(
            rmb_data[DYNAMIXEL_POSITION_LIMIT_MIN][:], dtype=np.float64
        )
        limit_max = np.asarray(
            rmb_data[DYNAMIXEL_POSITION_LIMIT_MAX][:], dtype=np.float64
        )

    return time, command, command_unclipped, measured, limit_min, limit_max


def load_dynamixel_effort_data(rmb_path):
    required_keys = [
        MEASURED_DYNAMIXEL_PRESENT_CURRENT_RAW,
        MEASURED_DYNAMIXEL_PRESENT_PWM_RAW,
        DYNAMIXEL_CURRENT_LIMIT_RAW,
        DYNAMIXEL_PWM_LIMIT_RAW,
    ]
    with RmbData(rmb_path) as rmb_data:
        missing_keys = [key for key in required_keys if key not in rmb_data]
        if missing_keys:
            raise ValueError(
                "RMB file does not contain Dynamixel effort debug keys: "
                + ", ".join(missing_keys)
            )
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        present_current = np.asarray(
            rmb_data[MEASURED_DYNAMIXEL_PRESENT_CURRENT_RAW][:], dtype=np.float64
        )
        present_pwm = np.asarray(
            rmb_data[MEASURED_DYNAMIXEL_PRESENT_PWM_RAW][:], dtype=np.float64
        )
        current_limit = np.asarray(
            rmb_data[DYNAMIXEL_CURRENT_LIMIT_RAW][:], dtype=np.float64
        )
        pwm_limit = np.asarray(rmb_data[DYNAMIXEL_PWM_LIMIT_RAW][:], dtype=np.float64)

    return time, present_current, present_pwm, current_limit, pwm_limit


def _marker_kwargs(markers):
    if not markers:
        return {}
    return {"marker": "o", "markersize": 2.5, "markeredgewidth": 0.0}


def plot_command_measured(
    time, command, measured, output=None, show_error=False, markers=False
):
    fig, axes = plt.subplots(4, 4, figsize=(18, 10), sharex=True)
    axes = axes.ravel()

    for joint_idx, ax in enumerate(axes):
        ax.plot(
            time,
            command[:, joint_idx],
            label="command",
            linewidth=1.5,
            **_marker_kwargs(markers),
        )
        ax.plot(
            time,
            measured[:, joint_idx],
            label="measured",
            linewidth=1.2,
            **_marker_kwargs(markers),
        )
        ax.set_title(JOINT_NAMES[joint_idx])
        ax.grid(True)

    axes[0].legend()
    fig.supxlabel("time [s]")
    fig.tight_layout()

    if output is not None:
        fig.savefig(output)
        print(f"Save plot: {output}")

    if show_error:
        fig_error, axes_error = plt.subplots(4, 4, figsize=(18, 10), sharex=True)
        axes_error = axes_error.ravel()
        error = measured - command
        for joint_idx, ax in enumerate(axes_error):
            ax.plot(
                time,
                error[:, joint_idx],
                linewidth=1.2,
                **_marker_kwargs(markers),
            )
            ax.set_title(f"{JOINT_NAMES[joint_idx]} measured - command")
            ax.grid(True)
        fig_error.supxlabel("time [s]")
        fig_error.tight_layout()

    if output is None or show_error:
        plt.show()


def plot_dynamixel_raw(
    time,
    command,
    command_unclipped,
    measured,
    limit_min,
    limit_max,
    output=None,
    markers=False,
):
    fig, axes = plt.subplots(6, 3, figsize=(18, 14), sharex=True)
    axes = axes.ravel()

    for motor_idx, ax in enumerate(axes):
        ax.plot(
            time,
            command[:, motor_idx],
            label="goal",
            linewidth=1.5,
            **_marker_kwargs(markers),
        )
        ax.plot(
            time,
            measured[:, motor_idx],
            label="present",
            linewidth=1.2,
            **_marker_kwargs(markers),
        )
        ax.plot(
            time,
            command_unclipped[:, motor_idx],
            label="goal unclipped",
            linewidth=1.0,
            linestyle="--",
            **_marker_kwargs(markers),
        )
        ax.axhline(limit_min[0, motor_idx], color="tab:red", linewidth=0.8)
        ax.axhline(limit_max[0, motor_idx], color="tab:red", linewidth=0.8)
        ax.set_title(DYNAMIXEL_NAMES[motor_idx])
        ax.grid(True)

    axes[0].legend()
    fig.supxlabel("time [s]")
    fig.tight_layout()

    if output is not None:
        fig.savefig(output)
        print(f"Save plot: {output}")
    else:
        plt.show()


def plot_dynamixel_effort(
    time,
    present_current,
    present_pwm,
    current_limit,
    pwm_limit,
    output=None,
    markers=False,
):
    fig, axes = plt.subplots(6, 3, figsize=(18, 14), sharex=True)
    axes = axes.ravel()

    for motor_idx, ax in enumerate(axes):
        ax.plot(
            time,
            np.abs(present_current[:, motor_idx]),
            label="abs present current/load",
            linewidth=1.2,
            linestyle="-",
            **_marker_kwargs(markers),
        )
        ax.plot(
            time,
            np.abs(present_pwm[:, motor_idx]),
            label="abs present PWM",
            linewidth=1.2,
            linestyle="--",
            **_marker_kwargs(markers),
        )
        if not np.isnan(current_limit[0, motor_idx]):
            ax.axhline(
                current_limit[0, motor_idx],
                color="tab:orange",
                linewidth=0.8,
                label="current limit" if motor_idx == 0 else None,
            )
        if not np.isnan(pwm_limit[0, motor_idx]):
            ax.axhline(
                pwm_limit[0, motor_idx],
                color="tab:red",
                linewidth=0.8,
                linestyle="--",
                label="PWM limit" if motor_idx == 0 else None,
            )
        ax.set_title(DYNAMIXEL_NAMES[motor_idx])
        ax.grid(True)

    axes[0].legend()
    fig.supxlabel("time [s]")
    fig.tight_layout()

    if output is not None:
        fig.savefig(output)
        print(f"Save plot: {output}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb", help="RMB directory or main.rmb.hdf5 path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path. If omitted, show an interactive window.",
    )
    parser.add_argument(
        "--show_error",
        action="store_true",
        help="Also plot measured - command error.",
    )
    parser.add_argument(
        "--dynamixel_raw",
        action="store_true",
        help="Plot Dynamixel raw goal/present values saved by follower replay.",
    )
    parser.add_argument(
        "--dynamixel_effort",
        action="store_true",
        help="Plot Dynamixel present current/load and PWM values.",
    )
    parser.add_argument(
        "--points",
        action="store_true",
        help="Plot sample points in addition to lines.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rmb_path = os.path.expanduser(args.rmb)
    output = None if args.output is None else os.path.expanduser(args.output)
    if args.dynamixel_raw:
        data = load_dynamixel_raw_data(rmb_path)
        plot_dynamixel_raw(*data, output=output, markers=args.points)
    elif args.dynamixel_effort:
        data = load_dynamixel_effort_data(rmb_path)
        plot_dynamixel_effort(*data, output=output, markers=args.points)
    else:
        time, command, measured = load_data(rmb_path)
        plot_command_measured(
            time, command, measured, output, args.show_error, markers=args.points
        )


if __name__ == "__main__":
    main()
