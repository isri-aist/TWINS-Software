import math
import re


class TwinArmsEskinMarkerPublisher:
    def __init__(
        self,
        topic,
        frame_id="eskin_map",
        node_name="twin_arms_eskin_marker_publisher",
        prox_min=0.0,
        prox_max=1.0,
        force_min=0.0,
        force_max=0.3,
        config_spacing_x=0.3,
        config_spacing_y=0.4,
        default_cell_radius=0.015,
        base_color=0.15,
        patch_offsets=None,
    ):
        try:
            import rclpy
            from geometry_msgs.msg import Point
            from rclpy.node import Node
            from std_msgs.msg import ColorRGBA
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] eSkin marker publishing requires "
                "ROS2 Python packages. Source the ROS2 setup file first."
            ) from e

        if not rclpy.ok():
            rclpy.init()
            self.initialized_rclpy = True
        else:
            self.initialized_rclpy = False

        self.rclpy = rclpy
        self.Point = Point
        self.ColorRGBA = ColorRGBA
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.node = Node(node_name)
        self.publisher = self.node.create_publisher(MarkerArray, topic, 1)
        self.topic = topic
        self.frame_id = frame_id
        self.prox_min = prox_min
        self.prox_max = prox_max
        self.force_min = force_min
        self.force_max = force_max
        self.config_spacing_x = config_spacing_x
        self.config_spacing_y = config_spacing_y
        self.default_cell_radius = default_cell_radius
        self.base_color = base_color
        self.patch_offsets = patch_offsets or {}
        self.last_config_order = []
        self.node.get_logger().info(f"Publish eSkin markers to {topic}")

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.initialized_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def publish(self, data, patch_names, patch_alpha=None):
        if data is None:
            return
        if patch_alpha is None:
            patch_alpha = {}

        stamp = self.node.get_clock().now().to_msg()
        marker_array = self.MarkerArray()
        config_order = [name for name in patch_names if name in data]
        if config_order != self.last_config_order:
            marker_array.markers.append(self._make_delete_all(stamp))
            self.last_config_order = config_order

        marker_id = 0
        offsets = self._config_offsets(config_order)
        for config_name in config_order:
            cells = data[config_name]["cells"]
            radius = self._auto_cell_radius(cells)
            pose_offset = offsets[config_name]
            alpha = patch_alpha.get(config_name, 1.0)
            for cell in cells:
                marker_array.markers.append(
                    self._make_hex_marker(
                        marker_id,
                        stamp,
                        config_name,
                        cell,
                        radius,
                        pose_offset,
                        alpha,
                    )
                )
                marker_id += 1

        self.publisher.publish(marker_array)

    def _make_delete_all(self, stamp):
        marker = self.Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.action = self.Marker.DELETEALL
        return marker

    def _make_hex_marker(
        self,
        marker_id,
        stamp,
        namespace,
        cell,
        radius,
        pose_offset,
        alpha,
    ):
        marker = self.Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = self.Marker.TRIANGLE_LIST
        marker.action = self.Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color = self._color_for_cell(cell, alpha)

        x, y = self._transform_xy(cell["position_x"], cell["position_y"], pose_offset)
        z = 0.0
        center = self.Point(x=x, y=y, z=z)
        corners = self._hex_points(x, y, z, radius, yaw=pose_offset[2])
        for idx in range(6):
            marker.points.append(center)
            marker.points.append(corners[idx])
            marker.points.append(corners[(idx + 1) % 6])
        return marker

    def _color_for_cell(self, cell, alpha=1.0):
        prox = self._normalize(cell["prox"], self.prox_min, self.prox_max)
        force = self._normalize(
            cell["force1"] + cell["force2"] + cell["force3"],
            self.force_min,
            self.force_max,
        )
        base = self.base_color
        return self.ColorRGBA(
            r=base + (1.0 - base) * force,
            g=base + (1.0 - base) * prox,
            b=base,
            a=alpha,
        )

    def _config_offsets(self, config_order):
        group_order = ["left", "chest", "right", "other"]
        row_indices = self._config_row_indices(config_order)
        offsets = {}
        for config_name in config_order:
            group_name = self._config_group(config_name)
            group_idx = group_order.index(group_name)
            row_idx = row_indices[config_name]
            xy_yaw = [
                row_idx * self.config_spacing_x,
                -group_idx * self.config_spacing_y,
                0.0,
            ]
            if config_name in self.patch_offsets:
                patch_offset = self.patch_offsets[config_name]
                xy_yaw = [
                    xy_yaw[0] + patch_offset[0],
                    xy_yaw[1] + patch_offset[1],
                    patch_offset[2],
                ]
            offsets[config_name] = xy_yaw
        return offsets

    def _config_row_indices(self, config_order):
        row_indices = {}
        for group_name in ("left", "chest", "right", "other"):
            group_configs = [
                config_name
                for config_name in config_order
                if self._config_group(config_name) == group_name
            ]
            group_configs = sorted(
                group_configs, key=lambda config_name: self._config_number(config_name)
            )
            for row_idx, config_name in enumerate(group_configs):
                row_indices[config_name] = row_idx
        return row_indices

    @staticmethod
    def _config_group(config_name):
        lower_name = config_name.lower()
        if "left" in lower_name:
            return "left"
        if "right" in lower_name:
            return "right"
        if "chest" in lower_name:
            return "chest"
        return "other"

    @staticmethod
    def _config_number(config_name):
        matches = re.findall(r"\d+", config_name)
        if len(matches) == 0:
            return math.inf, config_name
        return int(matches[-1]), config_name

    def _auto_cell_radius(self, cells):
        positions = [(cell["position_x"], cell["position_y"]) for cell in cells]
        min_dist = None
        for idx0, pos0 in enumerate(positions):
            for pos1 in positions[idx0 + 1 :]:
                dist = math.hypot(pos0[0] - pos1[0], pos0[1] - pos1[1])
                if dist > 1.0e-9 and (min_dist is None or dist < min_dist):
                    min_dist = dist
        if min_dist is None:
            return self.default_cell_radius
        return 0.45 * min_dist

    @staticmethod
    def _transform_xy(x, y, xy_yaw):
        cos_yaw = math.cos(xy_yaw[2])
        sin_yaw = math.sin(xy_yaw[2])
        return (
            xy_yaw[0] + cos_yaw * x - sin_yaw * y,
            xy_yaw[1] + sin_yaw * x + cos_yaw * y,
        )

    def _hex_points(self, x, y, z, radius, yaw=0.0):
        angle_offset = math.pi / 6.0 + math.pi / 2.0
        return [
            self.Point(
                x=x + radius * math.cos(yaw + angle_offset + idx * math.pi / 3.0),
                y=y + radius * math.sin(yaw + angle_offset + idx * math.pi / 3.0),
                z=z,
            )
            for idx in range(6)
        ]

    @staticmethod
    def _normalize(value, min_value, max_value):
        if max_value <= min_value:
            return 0.0
        return max(0.0, min((value - min_value) / (max_value - min_value), 1.0))
