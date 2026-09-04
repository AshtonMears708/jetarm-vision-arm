"""
JetArm ROS 2 Arm Control — Python API Starter
Prereq: ros2 launch sdk jetarm_sdk.launch.py must be running
Usage: python3 jetarm_control.py
"""

import rclpy
from rclpy.node import Node

# Servo topic message
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition

# Kinematics service messages
from kinematics_msgs.srv import SetJointValue, SetRobotPose


class JetArmController(Node):
    def __init__(self):
        super().__init__('jetarm_controller')

        # ── Publisher: direct servo control ──────────────────────────────────
        self.servo_pub = self.create_publisher(
            ServosPosition,
            '/ros_robot_controller/bus_servo/set_position',
            10
        )

        # ── Service clients: FK and IK ────────────────────────────────────────
        self.fk_client = self.create_client(
            SetJointValue,
            '/kinematics/set_joint_value_target'
        )
        self.ik_client = self.create_client(
            SetRobotPose,
            '/kinematics/set_pose_target'
        )

        self.get_logger().info('JetArmController node started.')

    # ── 1. Direct servo control ───────────────────────────────────────────────
    def set_servo(self, servo_id: int, position: int):
        """
        Move a single servo to a target position.
        position: typically 0–1000 (pulse-width units, ~500 = centre)
        """
        msg = ServosPosition()
        servo = ServoPosition()
        servo.id = servo_id
        servo.position = position
        msg.position = [servo]
        self.servo_pub.publish(msg)
        self.get_logger().info(f'Servo {servo_id} → {position}')

    def set_servos(self, id_position_pairs: list[tuple[int, int]]):
        """
        Move multiple servos simultaneously.
        id_position_pairs: [(id, position), ...]
        Example: [(1, 500), (2, 400), (3, 600)]
        """
        msg = ServosPosition()
        servos = []
        for sid, pos in id_position_pairs:
            s = ServoPosition()
            s.id = sid
            s.position = pos
            servos.append(s)
        msg.position = servos
        self.servo_pub.publish(msg)
        self.get_logger().info(f'Servos set: {id_position_pairs}')

    # ── 2. Forward kinematics (joint-space) ───────────────────────────────────
    def set_joint_values(self, joint_values: list[float], timeout: float = 5.0):
        """
        Move arm to joint-space target.
        joint_values: list of 5 floats, e.g. [500.0, 400.0, 300.0, 400.0, 500.0]
        """
        if not self.fk_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('FK service not available!')
            return None

        request = SetJointValue.Request()
        request.joint_value = [float(v) for v in joint_values]

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        self.get_logger().info(f'FK result: {result}')
        return result

    # ── 3. Inverse kinematics (Cartesian-space) ───────────────────────────────
    def set_pose_target(
        self,
        position: list[float],     # [x, y, z] in metres, e.g. [0.35, 0.0, 0.24]
        pitch: float = 10.0,
        pitch_range: list[float] = [-180.0, 180.0],
        resolution: int = 1,
        timeout: float = 5.0
    ):
        """
        Move arm end-effector to a Cartesian target via IK.
        position:    [x, y, z] in metres
        pitch:       desired pitch of the end-effector (degrees)
        pitch_range: allowable pitch search range [min, max]
        resolution:  IK search resolution (1 = finest)
        """
        if not self.ik_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('IK service not available!')
            return None

        request = SetRobotPose.Request()
        request.position = [float(v) for v in position]
        request.pitch = float(pitch)
        request.pitch_range = [float(v) for v in pitch_range]
        request.resolution = resolution

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        self.get_logger().info(f'IK result: {result}')
        return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    arm = JetArmController()

    # Example 1 — Move servo 1 to centre
    arm.set_servo(servo_id=1, position=500)

    # Example 2 — Move multiple servos at once
    arm.set_servos([(1, 500), (2, 400), (3, 600), (4, 500), (5, 500)])

    # Example 3 — Forward kinematics (joint values)
    arm.set_joint_values([500.0, 400.0, 300.0, 400.0, 500.0])

    # Example 4 — Inverse kinematics (Cartesian target)
    arm.set_pose_target(
        position=[0.35, 0.0, 0.24],
        pitch=10.0,
        pitch_range=[-180.0, 180.0]
    )

    arm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
