class JointStatePublisherMixin:
    def init_joint_state_publishers(self):
        self.joint_state_node = None
        self.joint_state_msg_class = None
        self.joint_state_publishers = {}
        self.initialized_rclpy = False

    def setup_joint_state_publishers(self, publisher_topics, node_name):
        if len(publisher_topics) == 0:
            return

        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as e:
            raise RuntimeError(
                f"[{self.__class__.__name__}] JointState publishing requires "
                "ROS2 Python packages. Source the ROS2 setup file first."
            ) from e

        if not rclpy.ok():
            rclpy.init()
            self.initialized_rclpy = True

        self.joint_state_msg_class = JointState
        self.joint_state_node = Node(node_name)
        for publisher_name, topic in publisher_topics.items():
            self.joint_state_publishers[publisher_name] = (
                self.joint_state_node.create_publisher(JointState, topic, 1)
            )
            self.joint_state_node.get_logger().info(
                f"Publish {publisher_name} joint states to {topic}"
            )

    def close_joint_state_publishers(self):
        if self.joint_state_node is not None:
            self.joint_state_node.destroy_node()
            self.joint_state_node = None
        if self.initialized_rclpy:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        self.joint_state_publishers = {}

    def publish_joint_states(self, publisher_name, joint_names, joint_pos):
        if not self.publish_joint_states_enabled:
            return
        publisher = self.joint_state_publishers[publisher_name]
        msg = self.joint_state_msg_class()
        msg.header.stamp = self.joint_state_node.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = joint_pos.tolist()
        publisher.publish(msg)
