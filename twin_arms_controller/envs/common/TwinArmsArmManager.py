import numpy as np
from robo_manip_baselines.common import ArmConfig, BodyManagerBase, DataKey


class TwinArmsArmManager(BodyManagerBase):
    SUPPORTED_DATA_KEYS = [
        DataKey.COMMAND_JOINT_POS,
        DataKey.COMMAND_GRIPPER_JOINT_POS,
    ]

    def reset(self):
        if self.env.unwrapped.read_only:
            joint_pos = self.env.unwrapped.cached_joint_pos
        else:
            joint_pos = self.env.unwrapped.cached_command_joint_pos
        self.arm_joint_pos = joint_pos[self.body_config.arm_joint_idxes].copy()
        self.gripper_joint_pos = joint_pos[self.body_config.gripper_joint_idxes].copy()

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_JOINT_POS:
            self.arm_joint_pos = command[self.body_config.arm_joint_idxes].copy()
            self.gripper_joint_pos = command[
                self.body_config.gripper_joint_idxes
            ].copy()
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            self.gripper_joint_pos = command[
                self.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ].copy()
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_command_data(self, key):
        if key == DataKey.COMMAND_JOINT_POS:
            return self.arm_joint_pos, self.gripper_joint_pos
        elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
            return self.gripper_joint_pos
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def draw_markers(self):
        pass


class TwinArmsArmConfig(ArmConfig):
    BodyManagerClass = TwinArmsArmManager

    def __init__(
        self,
        arm_joint_idxes,
        gripper_joint_idxes,
        gripper_joint_idxes_in_gripper_joint_pos,
    ):
        super().__init__(
            arm_urdf_path="",
            arm_root_pose=None,
            ik_eef_joint_id=0,
            arm_joint_idxes=arm_joint_idxes,
            gripper_joint_idxes=gripper_joint_idxes,
            gripper_joint_idxes_in_gripper_joint_pos=gripper_joint_idxes_in_gripper_joint_pos,
            eef_idx=None,
            init_arm_joint_pos=np.zeros(len(arm_joint_idxes)),
            init_gripper_joint_pos=np.zeros(len(gripper_joint_idxes)),
        )
