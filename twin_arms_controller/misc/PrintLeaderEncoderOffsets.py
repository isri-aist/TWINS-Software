#!/usr/bin/env python3

import argparse
import os

import numpy as np
from robo_manip_baselines.common import DataKey, RmbData

from twin_arms_controller.envs.leader.RealTwinArmsLeaderEnv import (
    RealTwinArmsLeaderEnv,
)


def format_array(name, values):
    value_str = ", ".join(f"{value:.17g}" for value in values)
    return (
        f"{name} = np.array(\n" f"    [{value_str}],\n" f"    dtype=np.float64,\n" f")"
    )


def load_joint_pos(rmb_path, data_key):
    with RmbData(rmb_path) as rmb_data:
        joint_pos = np.asarray(rmb_data[data_key][:], dtype=np.float64)

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 16:
        raise ValueError(
            f"Expected {data_key} shape to be (N, 16), but got {joint_pos.shape}."
        )
    return joint_pos


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb", help="RMB directory or main.rmb.hdf5 path")
    parser.add_argument(
        "--data_key",
        default=DataKey.MEASURED_JOINT_POS,
        choices=[DataKey.MEASURED_JOINT_POS, DataKey.COMMAND_JOINT_POS],
        help="RMB data key used to compute the zero-pose average.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rmb_path = os.path.expanduser(args.rmb)
    joint_pos = load_joint_pos(rmb_path, args.data_key)

    mean_joint_pos = np.mean(joint_pos, axis=0)
    current_offset_left = RealTwinArmsLeaderEnv.encoder_offset_left
    current_offset_right = RealTwinArmsLeaderEnv.encoder_offset_right
    current_offset = np.concatenate([current_offset_left, current_offset_right])
    new_offset = current_offset - mean_joint_pos

    print(f"# rmb: {rmb_path}")
    print(f"# data_key: {args.data_key}")
    print(f"# num_samples: {joint_pos.shape[0]}")
    print("# mean_joint_pos:")
    print(format_array("mean_joint_pos", mean_joint_pos))
    print()
    print("# Copy the following arrays into RealTwinArmsLeaderEnv.py:")
    print(format_array("encoder_offset_left", new_offset[:8]))
    print(format_array("encoder_offset_right", new_offset[8:]))


if __name__ == "__main__":
    main()
