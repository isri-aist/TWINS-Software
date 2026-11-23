import json

import yaml


def load_eskin_layout(path):
    with open(path, "r") as f:
        layout = json.load(f)

    if layout.get("format") != "twin_arms_eskin_layout_v1":
        raise ValueError(f"Unsupported eSkin layout format: {layout.get('format')}")
    if "patch_names" not in layout or "patches" not in layout:
        raise ValueError("eSkin layout must contain patch_names and patches.")

    missing_patch_names = [
        patch_name
        for patch_name in layout["patch_names"]
        if patch_name not in layout["patches"]
    ]
    if missing_patch_names:
        raise ValueError(
            "eSkin layout patches not found: " + ", ".join(missing_patch_names)
        )

    return layout


def save_eskin_layout(path, patch_names, result):
    layout = {
        "format": "twin_arms_eskin_layout_v1",
        "patch_names": list(patch_names),
        "patches": {},
    }
    for patch_name in patch_names:
        patch_data = result[patch_name]
        layout["patches"][patch_name] = {
            "frame_id": patch_data.get("frame_id", ""),
            "cell_count": int(patch_data["cell_count"]),
            "cells": [_make_layout_cell(cell) for cell in patch_data["cells"]],
        }

    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
        f.write("\n")


def layout_to_marker_data(layout):
    return {
        patch_name: {
            "frame_id": layout["patches"][patch_name].get("frame_id", ""),
            "cell_count": int(layout["patches"][patch_name]["cell_count"]),
            "cells": [cell.copy() for cell in layout["patches"][patch_name]["cells"]],
        }
        for patch_name in layout["patch_names"]
    }


def load_eskin_patch_offsets(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if "patch_offsets" in config:
        config = config["patch_offsets"]
    return {
        patch_name: _parse_offset(offset, patch_name)
        for patch_name, offset in config.items()
    }


def get_layout_cell_counts(layout):
    return tuple(
        int(layout["patches"][patch_name]["cell_count"])
        for patch_name in layout["patch_names"]
    )


def _make_layout_cell(cell):
    return {
        "cell_id": int(cell["cell_id"]),
        "patch_id": int(cell["patch_id"]),
        "segment_id": int(cell["segment_id"]),
        "position_x": float(cell["position_x"]),
        "position_y": float(cell["position_y"]),
        "position_z": float(cell["position_z"]),
        "orientation_x": float(cell.get("orientation_x", 0.0)),
        "orientation_y": float(cell.get("orientation_y", 0.0)),
        "orientation_z": float(cell.get("orientation_z", 0.0)),
        "orientation_w": float(cell.get("orientation_w", 1.0)),
        "force1": 0.0,
        "force2": 0.0,
        "force3": 0.0,
        "prox": 0.0,
        "acc_x": 0.0,
        "acc_y": 0.0,
        "acc_z": 0.0,
    }


def _parse_offset(offset, patch_name):
    if isinstance(offset, list):
        if len(offset) != 3:
            raise ValueError(f"offset of {patch_name} must be [x, y, yaw]: {offset}")
        return [float(value) for value in offset]
    raise ValueError(f"offset of {patch_name} must be [x, y, yaw]: {offset}")
