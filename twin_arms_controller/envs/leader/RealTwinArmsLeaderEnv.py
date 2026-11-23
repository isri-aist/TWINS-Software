import concurrent.futures

import numpy as np
from gymnasium.spaces import Box, Dict
from robo_manip_baselines.common import DataKey
from robo_manip_baselines.envs.real.RealEnvBase import RealEnvBase

from ..common import (
    JointStatePublisherMixin,
    TwinArmsArmConfig,
    TwinArmsEskinMarkerPublisher,
    TwinArmsEskinReceiver,
    TwinArmsJointInputDevice,
)
from .LeaderEncoderSerial import LeaderEncoderSerial


class RealTwinArmsLeaderEnv(JointStatePublisherMixin, RealEnvBase):
    joint_names = [
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

    gripper_count_to_pos = 0.04 / 300.0

    encoder_sign_left = np.array(
        [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
        dtype=np.float64,
    )
    encoder_sign_right = np.array(
        [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0],
        dtype=np.float64,
    )
    encoder_offset_left = np.array(
        [
            1.1719613219446299,
            1.8898643306751099,
            -1.4251176351969828,
            2.3009711818284617,
            -2.7557830294647307,
            1.8960002538266527,
            0.68108746982122481,
            0,
        ],
        dtype=np.float64,
    )
    encoder_offset_right = np.array(
        [
            1.8162332528565992,
            1.564660403643354,
            1.9634954084936207,
            -1.000155473701438,
            2.1271200258680891,
            1.7180584824319178,
            0.71790300873048007,
            0,
        ],
        dtype=np.float64,
    )

    action_space = Box(
        low=np.array([-2 * np.pi] * 7 + [-0.04] + [-2 * np.pi] * 7 + [-0.04]),
        high=np.array([2 * np.pi] * 7 + [0.0] + [2 * np.pi] * 7 + [0.0]),
        dtype=np.float64,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        leader_device_left,
        leader_device_right,
        camera_ids=None,
        publish_joint_states=False,
        joint_state_topic="/joint_states",
        eskin_endpoint=None,
        eskin_patch_names=None,
        eskin_timeout=1.0,
        publish_eskin_markers=False,
        eskin_marker_topic="/twin_arms/eskin/markers",
        eskin_marker_frame_id="eskin_map",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dt = 0.1  # [s]

        self.read_only = True
        self.leader_device_left = leader_device_left
        self.leader_device_right = leader_device_right
        self.encoder_left = LeaderEncoderSerial(leader_device_left)
        self.encoder_right = LeaderEncoderSerial(leader_device_right)
        self.eskin_enabled = eskin_endpoint is not None
        self.eskin_receiver = None
        self.eskin_marker_publisher = None
        self.cached_eskin = None

        if self.eskin_enabled:
            self.eskin_receiver = TwinArmsEskinReceiver(
                eskin_endpoint, eskin_patch_names, timeout=eskin_timeout
            )
            self.cached_eskin = self.eskin_receiver.read()
            if publish_eskin_markers:
                self.eskin_marker_publisher = TwinArmsEskinMarkerPublisher(
                    eskin_marker_topic,
                    frame_id=eskin_marker_frame_id,
                    node_name="real_twin_arms_leader_eskin_marker_publisher",
                )
                self._publish_eskin_markers()

        self.body_config_list = [
            TwinArmsArmConfig(
                arm_joint_idxes=np.arange(0, 7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
            ),
            TwinArmsArmConfig(
                arm_joint_idxes=np.arange(8, 15),
                gripper_joint_idxes=np.array([15]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
            ),
        ]

        self.cached_joint_pos = np.zeros(16, dtype=np.float64)
        self.prev_joint_pos = self.cached_joint_pos.copy()
        self.cached_joint_vel = np.zeros(16, dtype=np.float64)
        if self.eskin_enabled:
            self.observation_space = Dict(
                {
                    **self.observation_space.spaces,
                    "eskin": Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.cached_eskin.shape,
                        dtype=np.float64,
                    ),
                }
            )
        self.publish_joint_states_enabled = publish_joint_states
        self.joint_state_topic = joint_state_topic
        self.init_joint_state_publishers()
        if self.publish_joint_states_enabled:
            self.setup_joint_state_publishers(
                {"measured": self.joint_state_topic},
                "real_twin_arms_leader_joint_state_publisher",
            )

        self.setup_realsense(camera_ids)

    @property
    def measured_keys_to_save(self):
        measured_keys = [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
        ]
        if self.eskin_enabled:
            measured_keys.append(DataKey.MEASURED_ESKIN)
        return measured_keys

    @property
    def command_keys_to_save(self):
        return [
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
        ]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        return [
            TwinArmsJointInputDevice(self, body_manager)
            for body_manager in motion_manager.body_manager_list
        ]

    def close(self):
        self.close_joint_state_publishers()
        if self.eskin_marker_publisher is not None:
            self.eskin_marker_publisher.close()
        if self.eskin_receiver is not None:
            self.eskin_receiver.close()
        self.encoder_left.close()
        self.encoder_right.close()

    def _reset_robot(self):
        self.cached_joint_pos = self._read_joint_pos()
        if self.eskin_enabled:
            self.cached_eskin = self.eskin_receiver.read()
            self._publish_eskin_markers()
        self.prev_joint_pos = self.cached_joint_pos.copy()
        self.cached_joint_vel = np.zeros(16, dtype=np.float64)
        self.publish_joint_states("measured", self.joint_names, self.cached_joint_pos)

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        self.prev_joint_pos = self.cached_joint_pos.copy()
        self.cached_joint_pos = self._read_joint_pos()
        if self.eskin_enabled:
            self.cached_eskin = self.eskin_receiver.read()
            self._publish_eskin_markers()
        self.cached_joint_vel = (self.cached_joint_pos - self.prev_joint_pos) / self.dt
        self.publish_joint_states("measured", self.joint_names, self.cached_joint_pos)

    def _get_obs(self):
        obs = {
            "joint_pos": self.cached_joint_pos.copy(),
            "joint_vel": self.cached_joint_vel.copy(),
        }
        if self.eskin_enabled:
            obs["eskin"] = self.cached_eskin.copy()
        return obs

    def modify_world(self, world_idx=None, cumulative_idx=None):
        return 0 if world_idx is None else world_idx

    def _publish_eskin_markers(self):
        if self.eskin_marker_publisher is None:
            return
        self.eskin_marker_publisher.publish(
            self.eskin_receiver.latest_result, self.eskin_receiver.patch_names
        )

    def _read_joint_pos(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(self.encoder_left.read_counts)
            right_future = executor.submit(self.encoder_right.read_counts)
            left_counts = left_future.result()
            right_counts = right_future.result()

        left = self._convert_counts(
            left_counts,
            self.encoder_sign_left,
            self.encoder_offset_left,
            self.gripper_count_to_pos,
        )
        right = self._convert_counts(
            right_counts,
            self.encoder_sign_right,
            self.encoder_offset_right,
            self.gripper_count_to_pos,
        )
        return np.concatenate([left, right]).astype(np.float64)

    @classmethod
    def _convert_counts(cls, counts, sign, offset, gripper_count_to_pos):
        joint_pos = np.array(
            [
                counts[0] * 2.0 * np.pi / 1024.0,
                counts[1] * 2.0 * np.pi / 1024.0,
                counts[2] * 2.0 * np.pi / 1024.0 / 15.5,
                counts[3] * 2.0 * np.pi / 1024.0,
                counts[4] * 2.0 * np.pi / 1024.0 / 5.7,
                counts[5] * 2.0 * np.pi / 1024.0,
                counts[6] * 2.0 * np.pi / 1024.0,
                counts[7] * gripper_count_to_pos,
            ],
            dtype=np.float64,
        )
        joint_pos = sign * joint_pos + offset
        return joint_pos
