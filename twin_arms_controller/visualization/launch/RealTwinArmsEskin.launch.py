from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = Path(__file__).resolve().parents[2]
    rviz_path = package_dir / "visualization" / "rviz" / "RealTwinArmsEskin.rviz"

    return LaunchDescription(
        [
            Node(
                package="rviz2",
                executable="rviz2",
                name="real_twin_arms_eskin_rviz",
                arguments=["-d", str(rviz_path)],
            ),
        ]
    )
