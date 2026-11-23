import concurrent.futures
import time
from dataclasses import dataclass

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
from .FollowerDynamixel import FollowerDynamixel


@dataclass(frozen=True)
class MotorSpec:
    port_key: str
    motor_id: int
    zero_raw: float
    sign: float
    raw_to_joint: float


class RealTwinArmsFollowerEnv(JointStatePublisherMixin, RealEnvBase):
    COMMAND_DYNAMIXEL_RAW_GOAL = "command_dynamixel_raw_goal"
    COMMAND_DYNAMIXEL_RAW_GOAL_UNCLIPPED = "command_dynamixel_raw_goal_unclipped"
    MEASURED_DYNAMIXEL_RAW_POS = "measured_dynamixel_raw_pos"
    DYNAMIXEL_POSITION_LIMIT_MIN = "dynamixel_position_limit_min"
    DYNAMIXEL_POSITION_LIMIT_MAX = "dynamixel_position_limit_max"
    MEASURED_DYNAMIXEL_PRESENT_CURRENT_RAW = "measured_dynamixel_present_current_raw"
    MEASURED_DYNAMIXEL_PRESENT_PWM_RAW = "measured_dynamixel_present_pwm_raw"
    DYNAMIXEL_CURRENT_LIMIT_RAW = "dynamixel_current_limit_raw"
    DYNAMIXEL_PWM_LIMIT_RAW = "dynamixel_pwm_limit_raw"

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

    x_raw_to_rad = 2.0 * np.pi / 4096.0
    p_raw_to_rad = 2.0 * np.pi / 1003846.0
    x_raw_to_rad_joint3 = x_raw_to_rad * 14.0 / 32.0
    double_motor_warning_threshold = np.deg2rad(10.0)
    x_raw_min = 0
    x_raw_max = 4095

    gripper_raw_to_pos = -0.08 / 4096.0

    # ID17 and ID27 are dual-mode slaves of ID16 and ID26. Use only the master
    # motors for both reading and actuation.
    joint_motor_specs = [
        [
            MotorSpec("left_p", 33, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("left_p", 32, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("left_x", 16, 1755, -1.0, x_raw_to_rad_joint3),
        ],
        [
            MotorSpec("left_p", 31, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("left_x", 15, 2048, -1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("left_x", 13, 2048, 1.0, x_raw_to_rad),
            MotorSpec("left_x", 14, 2048, -1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("left_x", 12, 2048, -1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("left_x", 11, 2103, 1.0, gripper_raw_to_pos),
        ],
        [
            MotorSpec("right_p", 43, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("right_p", 42, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("right_x", 26, 2340, 1.0, x_raw_to_rad_joint3),
        ],
        [
            MotorSpec("right_p", 41, 0, -1.0, p_raw_to_rad),
        ],
        [
            MotorSpec("right_x", 25, 2048, 1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("right_x", 23, 2048, 1.0, x_raw_to_rad),
            MotorSpec("right_x", 24, 2048, -1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("right_x", 22, 2048, -1.0, x_raw_to_rad),
        ],
        [
            MotorSpec("right_x", 21, 1700, 1.0, gripper_raw_to_pos),
        ],
    ]

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
        follower_device_left_x,
        follower_device_left_p,
        follower_device_right_x,
        follower_device_right_p,
        read_only=True,
        baudrate=57600,
        move_to_init_on_reset=False,
        init_joint_pos=None,
        command_joint_pos_offset=None,
        dynamixel_config=None,
        parallel_goal_write=False,
        save_follower_debug_data=False,
        camera_ids=None,
        publish_joint_states=False,
        joint_state_topic="/joint_states",
        command_joint_state_topic="/command_joint_states",
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

        self.read_only = read_only
        self.baudrate = baudrate
        self.move_to_init_on_reset = move_to_init_on_reset
        self.init_joint_pos = self._make_init_joint_pos(init_joint_pos)
        self.command_joint_pos_offset = self._make_command_joint_pos_offset(
            command_joint_pos_offset
        )
        self.dynamixel_config = self._make_dynamixel_config(dynamixel_config)
        self.parallel_goal_write = parallel_goal_write
        self.save_follower_debug_data = save_follower_debug_data
        self.actuation_setup_done = False
        self.init_joint_state_publishers()
        self.eskin_enabled = eskin_endpoint is not None
        self.eskin_receiver = None
        self.eskin_marker_publisher = None
        self.cached_eskin = None

        self.dynamixel_ports = {}
        try:
            self.dynamixel_ports = {
                "left_x": FollowerDynamixel(
                    follower_device_left_x,
                    [11, 12, 13, 14, 15, 16, 17],
                    "x",
                    baudrate,
                    enable_present_effort_read=save_follower_debug_data,
                ),
                "left_p": FollowerDynamixel(
                    follower_device_left_p,
                    [31, 32, 33],
                    "p",
                    baudrate,
                    enable_present_effort_read=save_follower_debug_data,
                ),
                "right_x": FollowerDynamixel(
                    follower_device_right_x,
                    [21, 22, 23, 24, 25, 26, 27],
                    "x",
                    baudrate,
                    enable_present_effort_read=save_follower_debug_data,
                ),
                "right_p": FollowerDynamixel(
                    follower_device_right_p,
                    [41, 42, 43],
                    "p",
                    baudrate,
                    enable_present_effort_read=save_follower_debug_data,
                ),
            }
        except Exception:
            self.close()
            raise

        if self.eskin_enabled:
            self.eskin_receiver = TwinArmsEskinReceiver(
                eskin_endpoint, eskin_patch_names, timeout=eskin_timeout
            )
            self.cached_eskin = self.eskin_receiver.read()
            if publish_eskin_markers:
                self.eskin_marker_publisher = TwinArmsEskinMarkerPublisher(
                    eskin_marker_topic,
                    frame_id=eskin_marker_frame_id,
                    node_name="real_twin_arms_follower_eskin_marker_publisher",
                )
                self._publish_eskin_markers()

        self.motor_debug_names = self._make_motor_debug_names()
        if self.save_follower_debug_data:
            self.motor_position_limit_min, self.motor_position_limit_max = (
                self._read_motor_position_limits()
            )
        else:
            self.motor_position_limit_min = np.full(
                len(self.motor_debug_names), np.nan, dtype=np.float64
            )
            self.motor_position_limit_max = np.full(
                len(self.motor_debug_names), np.nan, dtype=np.float64
            )

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
        self.cached_command_joint_pos = self.cached_joint_pos.copy()
        self.double_motor_error = np.zeros(16, dtype=np.float64)
        self.last_double_motor_warning_time = 0.0
        self.cached_measured_motor_raw_pos = np.zeros(
            len(self.motor_debug_names), dtype=np.float64
        )
        self.cached_command_motor_raw_goal = np.zeros(
            len(self.motor_debug_names), dtype=np.float64
        )
        self.cached_command_motor_raw_goal_unclipped = np.zeros(
            len(self.motor_debug_names), dtype=np.float64
        )
        self.cached_measured_motor_present_current = np.zeros(
            len(self.motor_debug_names), dtype=np.float64
        )
        self.cached_measured_motor_present_pwm = np.zeros(
            len(self.motor_debug_names), dtype=np.float64
        )
        self.motor_current_limit = np.full(
            len(self.motor_debug_names), np.nan, dtype=np.float64
        )
        self.motor_pwm_limit = np.full(
            len(self.motor_debug_names), np.nan, dtype=np.float64
        )
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
        self.command_joint_state_topic = command_joint_state_topic
        if self.publish_joint_states_enabled:
            publisher_topics = {"measured": self.joint_state_topic}
            if not self.read_only:
                publisher_topics["command"] = self.command_joint_state_topic
            self.setup_joint_state_publishers(
                publisher_topics, "real_twin_arms_follower_joint_state_publisher"
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
        if not self.read_only:
            try:
                self._disable_torque_all()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Failed to disable torque: {e}")
        self.close_joint_state_publishers()
        if self.eskin_marker_publisher is not None:
            self.eskin_marker_publisher.close()
        if self.eskin_receiver is not None:
            self.eskin_receiver.close()
        for dynamixel_port in self.dynamixel_ports.values():
            dynamixel_port.close()

    def _reset_robot(self):
        if self.read_only:
            self._disable_torque_all()
        elif not self.actuation_setup_done:
            self._setup_actuation()
            self.actuation_setup_done = True

        self.cached_joint_pos = self._read_joint_pos()
        if self.eskin_enabled:
            self.cached_eskin = self.eskin_receiver.read()
            self._publish_eskin_markers()
        self.prev_joint_pos = self.cached_joint_pos.copy()
        self.cached_joint_vel = np.zeros(16, dtype=np.float64)
        if self.move_to_init_on_reset:
            self.cached_command_joint_pos = self._clip_action(self.init_joint_pos)
        else:
            self.cached_command_joint_pos = self.cached_joint_pos.copy()

        if not self.read_only:
            self._write_joint_pos(self.cached_command_joint_pos)

        self.publish_joint_states("measured", self.joint_names, self.cached_joint_pos)
        if not self.read_only:
            self.publish_joint_states(
                "command", self.joint_names, self.cached_command_joint_pos
            )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        if not self.read_only:
            self.cached_command_joint_pos = self._clip_action(
                np.asarray(action, dtype=np.float64) + self.command_joint_pos_offset
            )
            self._write_joint_pos(self.cached_command_joint_pos)

        self.prev_joint_pos = self.cached_joint_pos.copy()
        self.cached_joint_pos = self._read_joint_pos()
        if self.eskin_enabled:
            self.cached_eskin = self.eskin_receiver.read()
            self._publish_eskin_markers()
        self.cached_joint_vel = (self.cached_joint_pos - self.prev_joint_pos) / self.dt
        self.publish_joint_states("measured", self.joint_names, self.cached_joint_pos)
        if not self.read_only:
            self.publish_joint_states(
                "command", self.joint_names, self.cached_command_joint_pos
            )

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

    def _disable_torque_all(self):
        for port in self.dynamixel_ports.values():
            port.disable_torque_all()

    def _enable_torque_all(self):
        for port in self.dynamixel_ports.values():
            port.enable_torque_all()

    def _setup_actuation(self):
        lr_configs = {
            "left": {
                "x_current_limit_motor_ids": [12, 13, 14, 15, 16, 17],
            },
            "right": {
                "x_current_limit_motor_ids": [22, 23, 24, 25, 26, 27],
            },
        }
        for side, lr_config in lr_configs.items():
            for series, series_config in self.dynamixel_config.items():
                port_key = f"{side}_{series}"
                config = series_config.copy()
                if series == "x":
                    config = {
                        **series_config,
                        "current_limit_motor_ids": lr_config[
                            "x_current_limit_motor_ids"
                        ],
                    }
                self.dynamixel_ports[port_key].setup_position_control(**config)
        if self.save_follower_debug_data:
            self.motor_position_limit_min, self.motor_position_limit_max = (
                self._read_motor_position_limits()
            )
            self.motor_current_limit, self.motor_pwm_limit = (
                self._get_motor_limit_debug_arrays()
            )
        self._enable_torque_all()

    def _read_joint_pos(self):
        raw_positions = {}
        for port_key, port in self.dynamixel_ports.items():
            raw_positions[port_key] = port.read_present_positions()
        if self.save_follower_debug_data:
            raw_efforts = {}
            for port_key, port in self.dynamixel_ports.items():
                raw_efforts[port_key] = port.read_present_efforts()
            self.cached_measured_motor_raw_pos = self._raw_positions_to_debug_array(
                raw_positions
            )
            (
                self.cached_measured_motor_present_current,
                self.cached_measured_motor_present_pwm,
            ) = self._raw_efforts_to_debug_arrays(raw_efforts)

        joint_pos = np.zeros(16, dtype=np.float64)
        self.double_motor_error = np.zeros(16, dtype=np.float64)
        for joint_idx, motor_specs in enumerate(self.joint_motor_specs):
            motor_joint_pos = np.array(
                [
                    self._motor_raw_to_joint(
                        raw_positions[motor_spec.port_key][motor_spec.motor_id],
                        motor_spec,
                    )
                    for motor_spec in motor_specs
                ],
                dtype=np.float64,
            )
            joint_pos[joint_idx] = float(np.mean(motor_joint_pos))
            if len(motor_joint_pos) > 1:
                self.double_motor_error[joint_idx] = float(
                    np.max(motor_joint_pos) - np.min(motor_joint_pos)
                )

        self._warn_large_double_motor_error()
        return joint_pos

    def _write_joint_pos(self, joint_pos):
        raw_goals_by_port = {port_key: {} for port_key in self.dynamixel_ports.keys()}
        if self.save_follower_debug_data:
            raw_goal_debug = self._joint_pos_to_motor_raw_debug(joint_pos)
            self.cached_command_motor_raw_goal = raw_goal_debug["raw_goal"]
            self.cached_command_motor_raw_goal_unclipped = raw_goal_debug[
                "raw_goal_unclipped"
            ]
        for joint_idx, motor_specs in enumerate(self.joint_motor_specs):
            for motor_spec in motor_specs:
                raw_goal = self._joint_to_motor_raw(joint_pos[joint_idx], motor_spec)
                raw_goals_by_port[motor_spec.port_key][motor_spec.motor_id] = raw_goal

        if self.parallel_goal_write:
            self._write_raw_goals_parallel(raw_goals_by_port)
        else:
            self._write_raw_goals_sequential(raw_goals_by_port)

    def _write_raw_goals_sequential(self, raw_goals_by_port):
        for port_key, raw_goals in raw_goals_by_port.items():
            self.dynamixel_ports[port_key].sync_write_goal_positions(raw_goals)

    def _write_raw_goals_parallel(self, raw_goals_by_port):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(raw_goals_by_port)
        ) as executor:
            futures = [
                executor.submit(
                    self.dynamixel_ports[port_key].sync_write_goal_positions,
                    raw_goals,
                )
                for port_key, raw_goals in raw_goals_by_port.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _clip_action(self, action):
        return np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )

    def _warn_large_double_motor_error(self):
        error_max = np.max(np.abs(self.double_motor_error))
        if error_max < self.double_motor_warning_threshold:
            return
        now = time.time()
        if now - self.last_double_motor_warning_time < 1.0:
            return
        self.last_double_motor_warning_time = now
        joint_idx = int(np.argmax(np.abs(self.double_motor_error)))
        print(
            f"[{self.__class__.__name__}] Large double-motor difference: "
            f"{self.joint_names[joint_idx]} "
            f"{self.double_motor_error[joint_idx]:.4f} rad "
            f"({np.rad2deg(self.double_motor_error[joint_idx]):.2f} deg)"
        )

    def get_motor_raw_debug_from_joint_pos(self, joint_pos):
        return self._joint_pos_to_motor_raw_debug(self._clip_action(joint_pos))

    def _make_motor_debug_names(self):
        motor_names = []
        for motor_specs in self.joint_motor_specs:
            for motor_spec in motor_specs:
                motor_names.append(f"{motor_spec.port_key}_id{motor_spec.motor_id}")
        return motor_names

    def _read_motor_position_limits(self):
        limits_by_port = {}
        for port_key, port in self.dynamixel_ports.items():
            limits_by_port[port_key] = port.read_position_limits()

        limit_min = []
        limit_max = []
        for motor_specs in self.joint_motor_specs:
            for motor_spec in motor_specs:
                limits = limits_by_port[motor_spec.port_key][motor_spec.motor_id]
                limit_min.append(limits["min"])
                limit_max.append(limits["max"])
        return (
            np.asarray(limit_min, dtype=np.float64),
            np.asarray(limit_max, dtype=np.float64),
        )

    def _raw_positions_to_debug_array(self, raw_positions):
        raw_debug = []
        for motor_specs in self.joint_motor_specs:
            for motor_spec in motor_specs:
                raw_debug.append(
                    raw_positions[motor_spec.port_key][motor_spec.motor_id]
                )
        return np.asarray(raw_debug, dtype=np.float64)

    def _raw_efforts_to_debug_arrays(self, raw_efforts):
        present_current = []
        present_pwm = []
        for motor_specs in self.joint_motor_specs:
            for motor_spec in motor_specs:
                effort = raw_efforts[motor_spec.port_key][motor_spec.motor_id]
                present_current.append(effort["current"])
                present_pwm.append(effort["pwm"])
        return (
            np.asarray(present_current, dtype=np.float64),
            np.asarray(present_pwm, dtype=np.float64),
        )

    def _get_motor_limit_debug_arrays(self):
        current_limit = []
        pwm_limit = []
        for motor_specs in self.joint_motor_specs:
            for motor_spec in motor_specs:
                limits = self.dynamixel_ports[motor_spec.port_key].motor_limit_values[
                    motor_spec.motor_id
                ]
                current_limit.append(limits["current_limit"])
                pwm_limit.append(limits["pwm_limit"])
        return (
            np.asarray(current_limit, dtype=np.float64),
            np.asarray(pwm_limit, dtype=np.float64),
        )

    def _joint_pos_to_motor_raw_debug(self, joint_pos):
        raw_goal = []
        raw_goal_unclipped = []
        for joint_idx, motor_specs in enumerate(self.joint_motor_specs):
            for motor_spec in motor_specs:
                unclipped = self._joint_to_motor_raw_unclipped(
                    joint_pos[joint_idx], motor_spec
                )
                clipped = self._clip_motor_raw(unclipped, motor_spec)
                raw_goal_unclipped.append(unclipped)
                raw_goal.append(float(int(round(clipped))))
        return {
            "raw_goal": np.asarray(raw_goal, dtype=np.float64),
            "raw_goal_unclipped": np.asarray(raw_goal_unclipped, dtype=np.float64),
        }

    @staticmethod
    def _joint_to_motor_raw_unclipped(joint_pos, motor_spec):
        return motor_spec.zero_raw + joint_pos / (
            motor_spec.sign * motor_spec.raw_to_joint
        )

    @staticmethod
    def _clip_motor_raw(raw, motor_spec):
        if motor_spec.port_key.endswith("_x"):
            return np.clip(
                raw,
                RealTwinArmsFollowerEnv.x_raw_min,
                RealTwinArmsFollowerEnv.x_raw_max,
            )
        return raw

    @staticmethod
    def _motor_raw_to_joint(raw_pos, motor_spec):
        return (
            motor_spec.sign * (raw_pos - motor_spec.zero_raw) * motor_spec.raw_to_joint
        )

    @staticmethod
    def _joint_to_motor_raw(joint_pos, motor_spec):
        raw = RealTwinArmsFollowerEnv._joint_to_motor_raw_unclipped(
            joint_pos, motor_spec
        )
        raw = RealTwinArmsFollowerEnv._clip_motor_raw(raw, motor_spec)
        return int(round(raw))

    @staticmethod
    def _make_dynamixel_config(_dynamixel_config):
        default_dynamixel_config = {
            "x": {
                "profile_velocity": 80,
                "profile_acceleration": 8,
                "pwm_limit": 650,
                "current_limit": 900,
                "position_limit_min": 0,
                "position_limit_max": 4095,
                "position_p_gain": 800,
                "motor_overrides": {},
            },
            "p": {
                "profile_velocity": 1800,
                "profile_acceleration": 1500,
                "pwm_limit": 1500,
                "current_limit": 4000,
                "position_limit_min": -501923,
                "position_limit_max": 501923,
                "motor_overrides": {},
            },
        }
        if _dynamixel_config is None:
            return default_dynamixel_config

        dynamixel_config = {
            series: series_config.copy()
            for series, series_config in default_dynamixel_config.items()
        }
        for series, series_config in _dynamixel_config.items():
            dynamixel_config[series].update(series_config)
            if "motor_overrides" in series_config:
                dynamixel_config[series]["motor_overrides"] = {
                    int(motor_id): motor_config
                    for motor_id, motor_config in series_config[
                        "motor_overrides"
                    ].items()
                }
        return dynamixel_config

    @staticmethod
    def _make_init_joint_pos(init_joint_pos):
        if init_joint_pos is None:
            return np.zeros(16, dtype=np.float64)
        init_joint_pos = np.asarray(init_joint_pos, dtype=np.float64)
        if init_joint_pos.shape != (16,):
            raise ValueError(
                "init_joint_pos must have 16 elements: "
                "[left_j1, ..., left_gripper, right_j1, ..., right_gripper]."
            )
        return init_joint_pos.copy()

    @classmethod
    def _make_command_joint_pos_offset(cls, command_joint_pos_offset):
        if command_joint_pos_offset is None:
            return np.zeros(len(cls.joint_names), dtype=np.float64)
        elif isinstance(command_joint_pos_offset, dict):
            unknown_names = [
                name for name in command_joint_pos_offset if name not in cls.joint_names
            ]
            if unknown_names:
                raise ValueError(
                    "Unknown command_joint_pos_offset names: "
                    + ", ".join(unknown_names)
                )
            offset = np.zeros(len(cls.joint_names), dtype=np.float64)
            for joint_idx, joint_name in enumerate(cls.joint_names):
                offset[joint_idx] = float(command_joint_pos_offset.get(joint_name, 0.0))
            return offset
        else:
            offset = np.asarray(command_joint_pos_offset, dtype=np.float64)
            if offset.shape != (len(cls.joint_names),):
                raise ValueError(
                    "command_joint_pos_offset must have 16 elements or be a dict keyed by "
                    "joint name."
                )
            return offset.copy()
