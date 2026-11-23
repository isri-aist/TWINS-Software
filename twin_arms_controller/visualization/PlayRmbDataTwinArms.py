#!/usr/bin/env python3

import argparse
import os
import pickle
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from robo_manip_baselines.common import DataKey, RmbData
from sensor_msgs.msg import JointState

from twin_arms_controller.common.data.EskinDataTransform import (
    normalize_raw_eskin_for_visualization,
)
from twin_arms_controller.envs.common import (
    MEASURED_ESKIN_KEY,
    TwinArmsEskinMarkerPublisher,
)
from twin_arms_controller.visualization.EskinLayout import (
    get_layout_cell_counts,
    layout_to_marker_data,
    load_eskin_layout,
    load_eskin_patch_offsets,
)

DEFAULT_URDF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "urdf", "TwinArmsLeader.urdf")
)
DEFAULT_ESKIN_PATCH_OFFSETS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "eskin_patch_offsets", "default.yaml")
)


class PlayRmbDataTwinArms(Node):
    def __init__(self, args, joint_names):
        super().__init__("play_rmb_data_twin_arms")

        self.args = args
        self.joint_names = joint_names
        self.joint_publisher = None
        self.command_joint_publisher = None
        self.eskin_publisher = None
        self.eskin_layout = None
        self.eskin_marker_data = None
        self.eskin_cell_counts = None
        self.eskin_patch_offsets = None
        self.policy_eskin_patch_names = None
        self.use_policy_eskin_normalization = False

        (
            self.time_seq,
            self.joint_pos_seq,
            self.command_joint_pos_seq,
            self.eskin_seq,
        ) = self.load_data(args)
        self.setup_joint()
        self.setup_eskin()

        self.time_idx = 0
        self.playback_speed = self.get_playback_speed(args)
        self.default_sleep_duration = self.get_default_sleep_duration()
        self.timer = self.create_timer(0.001, self.on_timer)

        self.get_logger().info(f"Loaded {args.rmb}: {len(self.time_seq)} samples.")
        self.get_logger().info(f"Playback speed: {self.playback_speed}.")

    def setup_joint(self):
        if self.args.no_joint:
            return
        if self.args.publish_command_joint_states:
            if self.joint_pos_seq is None:
                raise KeyError(
                    f"RMB data does not contain {DataKey.MEASURED_JOINT_POS}."
                )
            if self.command_joint_pos_seq is None:
                raise KeyError(
                    f"RMB data does not contain {DataKey.COMMAND_JOINT_POS}."
                )
        elif self.joint_pos_seq is None:
            raise KeyError(f"RMB data does not contain {DataKey.MEASURED_JOINT_POS}.")
        if len(self.joint_names) != self.joint_pos_seq.shape[1]:
            raise ValueError(
                f"The number of joint names ({len(self.joint_names)}) must match "
                f"the joint data dimension ({self.joint_pos_seq.shape[1]})."
            )
        self.joint_publisher = self.create_publisher(JointState, "joint_states", 10)
        if self.args.publish_command_joint_states:
            self.command_joint_publisher = self.create_publisher(
                JointState, "command_joint_states", 10
            )

    def setup_eskin(self):
        if self.args.no_eskin:
            return
        if self.eskin_seq is None:
            raise KeyError(f"RMB data does not contain {MEASURED_ESKIN_KEY}.")
        self.eskin_layout = load_eskin_layout(self.args.eskin_layout)
        self.eskin_marker_data = layout_to_marker_data(self.eskin_layout)
        self.eskin_cell_counts = get_layout_cell_counts(self.eskin_layout)
        self.eskin_patch_offsets = load_eskin_patch_offsets(
            self.args.eskin_patch_offsets
        )
        expected_dim = 2 * sum(self.eskin_cell_counts)
        if self.eskin_seq.shape[1] != expected_dim:
            raise ValueError(
                f"measured_eskin dim mismatch: {self.eskin_seq.shape[1]} != "
                f"{expected_dim} from eSkin layout."
            )
        if self.args.model_meta_info is not None:
            with open(os.path.expanduser(self.args.model_meta_info), "rb") as f:
                model_meta_info = pickle.load(f)
            eskin_transform = model_meta_info["state"].get("eskin")
            if eskin_transform is None:
                raise ValueError(
                    "model_meta_info does not contain state['eskin'] transform."
                )
            self.eskin_seq = normalize_raw_eskin_for_visualization(
                self.eskin_seq, eskin_transform
            )
            self.policy_eskin_patch_names = set(eskin_transform["patch_names"])
            self.use_policy_eskin_normalization = True

        self.eskin_publisher = TwinArmsEskinMarkerPublisher(
            self.args.eskin_topic,
            frame_id=self.args.eskin_frame_id,
            node_name="play_rmb_data_twin_arms_eskin_marker_publisher",
            force_max=1.0 if self.use_policy_eskin_normalization else 0.3,
            patch_offsets=self.eskin_patch_offsets,
        )

    @staticmethod
    def load_data(args):
        with RmbData(args.rmb) as rmb_data:
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
            joint_pos_seq = None
            command_joint_pos_seq = None
            eskin_seq = None
            if not args.no_joint and DataKey.MEASURED_JOINT_POS in rmb_data:
                joint_pos_seq = np.asarray(
                    rmb_data[DataKey.MEASURED_JOINT_POS][:], dtype=np.float64
                )
            if (
                args.publish_command_joint_states
                and DataKey.COMMAND_JOINT_POS in rmb_data
            ):
                command_joint_pos_seq = np.asarray(
                    rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64
                )
            if not args.no_eskin and MEASURED_ESKIN_KEY in rmb_data:
                eskin_seq = np.asarray(
                    rmb_data[MEASURED_ESKIN_KEY][:], dtype=np.float64
                )
        return time_seq, joint_pos_seq, command_joint_pos_seq, eskin_seq

    def on_timer(self):
        stamp = self.get_clock().now().to_msg()
        self.publish_joint(stamp)
        self.publish_eskin()

        sleep_duration = self.get_sleep_duration()
        if self.time_idx == len(self.time_seq) - 1:
            if not self.args.loop:
                rclpy.shutdown()
                return
            self.time_idx = 0
        else:
            self.time_idx += 1

        if sleep_duration > 0.0:
            time.sleep(sleep_duration)

    def get_sleep_duration(self):
        if self.time_idx < len(self.time_seq) - 1:
            data_duration = (
                self.time_seq[self.time_idx + 1] - self.time_seq[self.time_idx]
            )
        else:
            data_duration = self.default_sleep_duration
        return max(0.0, float(data_duration) / self.playback_speed)

    def get_default_sleep_duration(self):
        if len(self.time_seq) < 2:
            return 0.0
        return float(np.median(np.diff(self.time_seq)))

    @staticmethod
    def get_playback_speed(args):
        if args.speed is not None and args.rate_scale is not None:
            raise ValueError("Specify only one of --speed or --rate_scale.")
        speed = args.speed if args.speed is not None else args.rate_scale
        if speed is None:
            speed = 1.0
        if speed <= 0.0:
            raise ValueError("Playback speed must be positive.")
        return float(speed)

    def publish_joint(self, stamp):
        if self.joint_publisher is None:
            return
        self.publish_single_joint_state(
            self.joint_publisher, stamp, self.joint_pos_seq[self.time_idx]
        )
        if self.command_joint_publisher is not None:
            self.publish_single_joint_state(
                self.command_joint_publisher,
                stamp,
                self.command_joint_pos_seq[self.time_idx],
            )

    def publish_single_joint_state(self, publisher, stamp, joint_pos):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = self.joint_names
        msg.position = joint_pos.tolist()
        publisher.publish(msg)

    def publish_eskin(self):
        if self.eskin_publisher is None:
            return
        data = self.make_eskin_marker_data(self.eskin_seq[self.time_idx])
        patch_alpha = self.make_eskin_patch_alpha()
        self.eskin_publisher.publish(
            data, self.eskin_layout["patch_names"], patch_alpha=patch_alpha
        )

    def make_eskin_patch_alpha(self):
        if self.policy_eskin_patch_names is None:
            return None
        return {
            patch_name: 1.0 if patch_name in self.policy_eskin_patch_names else 0.5
            for patch_name in self.eskin_layout["patch_names"]
        }

    def make_eskin_marker_data(self, eskin):
        num_cells = sum(self.eskin_cell_counts)
        prox_values = eskin[:num_cells]
        force_values = eskin[num_cells:]

        data = {}
        offset = 0
        for patch_name, cell_count in zip(
            self.eskin_layout["patch_names"], self.eskin_cell_counts
        ):
            patch_data = self.eskin_marker_data[patch_name]
            cells = []
            for cell_idx, cell in enumerate(patch_data["cells"]):
                new_cell = cell.copy()
                value_idx = offset + cell_idx
                new_cell["prox"] = float(prox_values[value_idx])
                force_sum = float(force_values[value_idx])
                new_cell["force1"] = force_sum
                new_cell["force2"] = 0.0
                new_cell["force3"] = 0.0
                cells.append(new_cell)
            data[patch_name] = {
                "frame_id": patch_data["frame_id"],
                "cell_count": cell_count,
                "cells": cells,
            }
            offset += cell_count
        return data

    def close(self):
        if self.eskin_publisher is not None:
            self.eskin_publisher.close()


def get_joint_names_from_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joint_names = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        if joint.find("mimic") is not None:
            continue
        joint_name = joint.attrib.get("name")
        if joint_name is not None:
            joint_names.append(joint_name)
    return joint_names


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb", help="RMB file or main.rmb.hdf5 path")
    parser.add_argument("--no_joint", action="store_true", help="Disable joint play.")
    parser.add_argument("--no_eskin", action="store_true", help="Disable eSkin play.")
    parser.add_argument("--urdf", default=DEFAULT_URDF_PATH)
    parser.add_argument("--joint_names", nargs="+", default=None)
    parser.add_argument(
        "--publish_command_joint_states",
        action="store_true",
        help=(
            "Publish measured_joint_pos to /joint_states and command_joint_pos "
            "to /command_joint_states."
        ),
    )
    parser.add_argument(
        "--eskin_layout",
        default=os.path.normpath(
            os.path.join(os.path.dirname(__file__), "eskin_layouts", "default.json")
        ),
    )
    parser.add_argument("--eskin_topic", default="/twin_arms/eskin/markers")
    parser.add_argument("--eskin_frame_id", default="base_link")
    parser.add_argument(
        "--eskin_patch_offsets",
        default=DEFAULT_ESKIN_PATCH_OFFSETS_PATH,
        help="YAML file that maps eSkin patch names to [x, y, yaw] marker offsets.",
    )
    parser.add_argument(
        "--model_meta_info",
        default=None,
        help="model_meta_info.pkl used to visualize normalized eSkin values.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback speed. For example, 2.0 plays all samples at double speed.",
    )
    parser.add_argument(
        "--rate_scale",
        type=float,
        default=None,
        help="Deprecated alias of --speed.",
    )
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.rmb = os.path.expanduser(args.rmb)

    if args.no_joint and args.no_eskin:
        raise ValueError("At least one of joint or eSkin playback must be enabled.")

    if args.joint_names is not None:
        joint_names = args.joint_names
    else:
        joint_names = get_joint_names_from_urdf(os.path.expanduser(args.urdf))

    rclpy.init()
    node = PlayRmbDataTwinArms(args, joint_names)
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
