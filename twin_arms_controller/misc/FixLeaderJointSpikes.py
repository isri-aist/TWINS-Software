#!/usr/bin/env python3

import argparse
import os

import numpy as np
from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.common.manager.DataManager import DataManager

LEFT_ARM = slice(0, 7)
LEFT_WHOLE_ARM = slice(0, 8)
LEFT_GRIPPER = 7
RIGHT_ARM = slice(8, 15)
RIGHT_WHOLE_ARM = slice(8, 16)
RIGHT_GRIPPER = 15
SIDE_INDEXES = [
    ("left", LEFT_ARM, LEFT_WHOLE_ARM),
    ("right", RIGHT_ARM, RIGHT_WHOLE_ARM),
]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", nargs="+", help="RMB file or directory path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Spike threshold for one-step arm joint position jump [rad].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print detected spikes without updating RMB data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rmb_paths = []
    for path in args.path:
        rmb_paths.extend(find_rmb_files(os.path.expanduser(path)))

    if not rmb_paths:
        raise RuntimeError("No RMB data found.")

    for rmb_path in rmb_paths:
        fix_leader_joint_spikes(
            rmb_path,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )


def fix_leader_joint_spikes(rmb_path, threshold, dry_run=False):
    mode = "r" if dry_run else "r+"
    with RmbData(rmb_path, mode=mode) as rmb_data:
        measured_joint_pos = np.asarray(rmb_data[DataKey.MEASURED_JOINT_POS][:])
        command_joint_pos = np.asarray(rmb_data[DataKey.COMMAND_JOINT_POS][:])
        time = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)

        fixed_joint_pos, spike_indices = fix_joint_pos(
            measured_joint_pos,
            threshold=threshold,
        )

        print(f"{rmb_path}:")
        for side in ("left", "right"):
            indices = spike_indices[side]
            if len(indices) == 0:
                print(f"  {side}: no spikes")
            else:
                times = [float(time[i]) for i in indices]
                print(f"  {side}: spikes at frames {indices}, time {times}")

        if dry_run:
            return

        if measured_joint_pos.shape != command_joint_pos.shape:
            raise ValueError(
                f"{rmb_path}: measured/command joint shape mismatch: "
                f"{measured_joint_pos.shape} vs {command_joint_pos.shape}"
            )

        write_if_exists(rmb_data, DataKey.MEASURED_JOINT_POS, fixed_joint_pos)
        write_if_exists(rmb_data, DataKey.COMMAND_JOINT_POS, fixed_joint_pos)

        left_gripper = fixed_joint_pos[:, [LEFT_GRIPPER]]
        right_gripper = fixed_joint_pos[:, [RIGHT_GRIPPER]]
        gripper_joint_pos = np.concatenate([left_gripper, right_gripper], axis=1)
        write_if_exists(rmb_data, DataKey.MEASURED_GRIPPER_JOINT_POS, gripper_joint_pos)
        write_if_exists(rmb_data, DataKey.COMMAND_GRIPPER_JOINT_POS, gripper_joint_pos)

        joint_vel = calc_joint_vel(fixed_joint_pos, time)
        write_if_exists(rmb_data, DataKey.MEASURED_JOINT_VEL, joint_vel)

        joint_pos_rel = calc_rel_data(DataKey.MEASURED_JOINT_POS_REL, fixed_joint_pos)
        write_if_exists(rmb_data, DataKey.MEASURED_JOINT_POS_REL, joint_pos_rel)
        write_if_exists(rmb_data, DataKey.COMMAND_JOINT_POS_REL, joint_pos_rel)

        gripper_joint_pos_rel = calc_rel_data(
            DataKey.MEASURED_GRIPPER_JOINT_POS_REL, gripper_joint_pos
        )
        write_if_exists(
            rmb_data, DataKey.MEASURED_GRIPPER_JOINT_POS_REL, gripper_joint_pos_rel
        )
        write_if_exists(
            rmb_data, DataKey.COMMAND_GRIPPER_JOINT_POS_REL, gripper_joint_pos_rel
        )

        print("  updated RMB data")


def fix_joint_pos(joint_pos, threshold):
    fixed = joint_pos.copy()
    spike_indices = {"left": [], "right": []}

    for i in range(1, len(fixed)):
        for side, arm_slice, whole_arm_slice in SIDE_INDEXES:
            jump = np.max(np.abs(joint_pos[i, arm_slice] - fixed[i - 1, arm_slice]))
            if jump <= threshold:
                continue

            fixed[i, whole_arm_slice] = fixed[i - 1, whole_arm_slice]
            spike_indices[side].append(i)

    return fixed, spike_indices


def calc_joint_vel(joint_pos, time):
    joint_vel = np.zeros_like(joint_pos)
    if len(joint_pos) < 2:
        return joint_vel

    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time must be strictly increasing to recalculate joint vel.")
    joint_vel[1:] = np.diff(joint_pos, axis=0) / dt[:, np.newaxis]
    return joint_vel


def calc_rel_data(rel_key, abs_data):
    abs_key = DataKey.get_abs_key(rel_key)
    data_manager = DataManager(env=None)
    data_manager.all_data_seq[abs_key] = []

    rel_data = []
    for data in abs_data:
        data_manager.all_data_seq[abs_key].append(data)
        rel_data.append(data_manager.calc_rel_data(rel_key))
    return np.asarray(rel_data, dtype=abs_data.dtype)


def write_if_exists(rmb_data, key, value):
    if key not in rmb_data.keys():
        return
    if rmb_data[key].shape != value.shape:
        raise ValueError(
            f"{rmb_data.path}: {key} shape mismatch: {rmb_data[key].shape} vs {value.shape}"
        )
    rmb_data[key][...] = value


if __name__ == "__main__":
    main()
