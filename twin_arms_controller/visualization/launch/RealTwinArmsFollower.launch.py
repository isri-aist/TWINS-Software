from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = Path(__file__).resolve().parents[2]
    urdf_path = package_dir / "urdf" / "TwinArmsFollower.urdf"
    rviz_path = package_dir / "visualization" / "rviz" / "RealTwinArmsFollower.rviz"

    # Resolve portable mesh paths for the ROS robot_description string.
    mesh_uri = (urdf_path.parent / "meshes").as_uri()
    robot_description = urdf_path.read_text().replace(
        'filename="meshes/', f'filename="{mesh_uri}/'
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="real_twin_arms_follower_measured_robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                remappings=[("joint_states", "/joint_states")],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="real_twin_arms_follower_command_robot_state_publisher",
                parameters=[
                    {
                        "robot_description": robot_description,
                        "frame_prefix": "command/",
                    }
                ],
                remappings=[("joint_states", "/command_joint_states")],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="real_twin_arms_follower_command_root_tf",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "base_link",
                    "--child-frame-id",
                    "command/base_link",
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="real_twin_arms_follower_rviz",
                arguments=["-d", str(rviz_path)],
            ),
        ]
    )
