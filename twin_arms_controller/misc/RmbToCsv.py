#!/usr/bin/env python3

import argparse
import csv
import os

import numpy as np
from robo_manip_baselines.common import DataKey, RmbData

from twin_arms_controller.common.data.EskinDataTransform import (
    MEASURED_ESKIN_KEY,
    get_eskin_meta_from_attrs,
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb", help="RMB directory or main.rmb.hdf5 path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path. Defaults to <rmb_name>.csv in the current directory.",
    )
    return parser.parse_args()


def make_default_output_path(rmb_path):
    path = os.path.normpath(rmb_path)
    basename = os.path.basename(path)
    if basename == "main.rmb.hdf5":
        basename = os.path.basename(os.path.dirname(path))
    if basename.endswith(".rmb"):
        basename = basename[: -len(".rmb")]
    return f"{basename}.csv"


def load_rmb_arrays(rmb_path):
    required_keys = [
        DataKey.TIME,
        DataKey.COMMAND_JOINT_POS,
        DataKey.MEASURED_JOINT_POS,
        MEASURED_ESKIN_KEY,
    ]
    with RmbData(rmb_path) as rmb_data:
        missing_keys = [key for key in required_keys if key not in rmb_data]
        if missing_keys:
            raise ValueError(
                "RMB data does not contain keys: " + ", ".join(missing_keys)
            )

        arrays = {
            DataKey.TIME: np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64),
            DataKey.COMMAND_JOINT_POS: np.asarray(
                rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64
            ),
            DataKey.MEASURED_JOINT_POS: np.asarray(
                rmb_data[DataKey.MEASURED_JOINT_POS][:], dtype=np.float64
            ),
            MEASURED_ESKIN_KEY: np.asarray(
                rmb_data[MEASURED_ESKIN_KEY][:], dtype=np.float64
            ),
        }
        eskin_column_names = make_eskin_column_names(
            rmb_data, arrays[MEASURED_ESKIN_KEY]
        )

    validate_arrays(arrays)
    return arrays, eskin_column_names


def validate_arrays(arrays):
    num_samples = arrays[DataKey.TIME].shape[0]
    if arrays[DataKey.TIME].ndim != 1:
        raise ValueError(f"time must be 1D, but got {arrays[DataKey.TIME].shape}.")

    for key in (DataKey.COMMAND_JOINT_POS, DataKey.MEASURED_JOINT_POS):
        data = arrays[key]
        if data.shape != (num_samples, len(JOINT_NAMES)):
            raise ValueError(
                f"{key} must have shape ({num_samples}, {len(JOINT_NAMES)}), "
                f"but got {data.shape}."
            )

    eskin = arrays[MEASURED_ESKIN_KEY]
    if eskin.ndim != 2 or eskin.shape[0] != num_samples:
        raise ValueError(
            f"{MEASURED_ESKIN_KEY} must have shape ({num_samples}, D), "
            f"but got {eskin.shape}."
        )


def make_eskin_column_names(rmb_data, eskin):
    patch_names, cell_counts = get_eskin_meta_from_attrs(rmb_data.attrs)
    total_cells = int(sum(cell_counts))
    if eskin.shape[1] != 2 * total_cells:
        raise ValueError(
            f"{MEASURED_ESKIN_KEY} dim mismatch: {eskin.shape[1]} != {2 * total_cells} "
            "from eSkin attrs."
        )

    prox_names = []
    force_names = []
    for patch_name, cell_count in zip(patch_names, cell_counts):
        for cell_idx in range(cell_count):
            prox_names.append(f"{MEASURED_ESKIN_KEY}_{patch_name}_prox_{cell_idx:03d}")
            force_names.append(
                f"{MEASURED_ESKIN_KEY}_{patch_name}_force_sum_{cell_idx:03d}"
            )
    return prox_names + force_names


def make_header(eskin_column_names):
    command_names = [
        f"{DataKey.COMMAND_JOINT_POS}_{joint_name}" for joint_name in JOINT_NAMES
    ]
    measured_names = [
        f"{DataKey.MEASURED_JOINT_POS}_{joint_name}" for joint_name in JOINT_NAMES
    ]
    return [DataKey.TIME] + command_names + measured_names + eskin_column_names


def write_csv(output_path, arrays, eskin_column_names):
    header = make_header(eskin_column_names)
    data = np.column_stack(
        [
            arrays[DataKey.TIME],
            arrays[DataKey.COMMAND_JOINT_POS],
            arrays[DataKey.MEASURED_JOINT_POS],
            arrays[MEASURED_ESKIN_KEY],
        ]
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"[RmbToCsv] Wrote {output_path}: {data.shape[0]} rows, {data.shape[1]} cols")


def main():
    args = parse_args()
    rmb_path = os.path.expanduser(args.rmb)
    output_path = (
        make_default_output_path(rmb_path)
        if args.output is None
        else os.path.expanduser(args.output)
    )
    arrays, eskin_column_names = load_rmb_arrays(rmb_path)
    write_csv(output_path, arrays, eskin_column_names)


if __name__ == "__main__":
    main()
