"""
JetArm Pick and Place API
Based on manufacturer's positioning_clamp.py

Usage:
    from arm_api import ArmAPI
    arm = ArmAPI()
    arm.home()
    arm.pick(x=0.23, y=0.006, z=0.015)
    arm.place(x=0.30, y=0.10, z=0.015)
"""

import time
import rclpy
from rclpy.node import Node
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import SetRobotPose
from servo_controller import bus_servo_control
from servo_controller_msgs.msg import ServosPosition
from std_srvs.srv import Trigger


class ArmAPI(Node):

    # Servo positions
    GRIPPER_OPEN   = 200   # Gripper open position
    GRIPPER_CLOSE  = 550   # Gripper close position
    GRIPPER_ID     = 10    # Gripper is servo 10
    PICK_PITCH     = 80    # Pitch angle for picking up

    # Safe home position
    HOME = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 550))

    def __init__(self):
        rclpy.init()
        super().__init__('arm_api')

        # Servo publisher — correct topic from manufacturer's code
        self.servos_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        # IK service client
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')

        # Wait for everything to be ready
        self.get_logger().info('Waiting for services...')
        init_client = self.create_client(Trigger, '/controller_manager/init_finish')
        init_client.wait_for_service()
        self.kinematics_client.wait_for_service()
        self.get_logger().info('ARM API ready.')

    # ── Servo helpers ─────────────────────────────────────────────────────────

    def set_servos(self, positions, speed=1.0):
        """
        Move servos to target positions.
        positions: tuple of (id, position) pairs e.g. ((1, 500), (2, 400))
        speed: time in seconds — higher = slower movement
        """
        bus_servo_control.set_servo_position(self.servos_pub, speed, positions)

    def open_gripper(self, speed=1.0):
        self.set_servos(((self.GRIPPER_ID, self.GRIPPER_OPEN),), speed)

    def close_gripper(self, speed=1.0):
        self.set_servos(((self.GRIPPER_ID, self.GRIPPER_CLOSE),), speed)

    def home(self, speed=1.5):
        """Move arm to safe home position."""
        self.set_servos(self.HOME, speed)
        time.sleep(speed + 0.5)

    # ── IK movement ───────────────────────────────────────────────────────────

    def move_to(self, x, y, z, speed=1.0):
        """
        Move end-effector to a Cartesian position using IK.
        x, y, z: target position in metres
        speed: movement time in seconds
        Returns True if successful, False if unreachable.
        """
        msg = set_pose_target([x, y, z], self.PICK_PITCH, [-90.0, 90.0], 1.0)
        future = self.kinematics_client.call_async(msg)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done() and future.result():
                res = future.result()
                if res.pulse:
                    self.get_logger().info(f'Moving to [{x}, {y}, {z}]')
                    self.set_servos(
                        ((1, res.pulse[0]), (2, res.pulse[1]), (3, res.pulse[2]),
                         (4, res.pulse[3]), (5, res.pulse[4])),
                        speed
                    )
                    time.sleep(speed + 0.5)
                    return True
                else:
                    self.get_logger().error(f'Position [{x}, {y}, {z}] is unreachable!')
                    return False

    # ── Pick and place ────────────────────────────────────────────────────────

    def pick(self, x, y, z=0.015, speed=1.0):
        """
        Pick up an object at the given position.
        x, y, z: position of the object in metres
        """
        self.get_logger().info(f'Picking at [{x}, {y}, {z}]')

        # Open gripper first
        self.open_gripper(speed)
        time.sleep(1.0)

        # Move above the object
        if not self.move_to(x, y, z + 0.05, speed):
            return False

        # Move down to the object
        if not self.move_to(x, y, z, speed):
            return False

        # Close gripper to grab
        self.close_gripper(speed)
        time.sleep(1.5)

        # Lift back up
        self.move_to(x, y, z + 0.05, speed)

        return True

    def place(self, x, y, z=0.015, speed=1.0):
        """
        Place the object at the given position.
        x, y, z: target drop position in metres
        """
        self.get_logger().info(f'Placing at [{x}, {y}, {z}]')

        # Move above the drop position
        if not self.move_to(x, y, z + 0.05, speed):
            return False

        # Lower down
        if not self.move_to(x, y, z, speed):
            return False

        # Release
        self.open_gripper(speed)
        time.sleep(1.0)

        # Return home
        self.home(speed)

        return True

    def pick_and_place(self, pick_pos, place_pos, speed=1.0):
        """
        Full pick and place sequence.
        pick_pos:  [x, y, z] of object to pick up
        place_pos: [x, y, z] of where to drop it
        """
        success = self.pick(*pick_pos, speed=speed)
        if success:
            self.place(*place_pos, speed=speed)
        else:
            self.get_logger().error('Pick failed, returning home.')
            self.home()


# ── Example usage ─────────────────────────────────────────────────────────────
def main():
    arm = ArmAPI()

    # Go home first
    arm.home()
    time.sleep(2)

    # Example: pick up a block at this position and place it somewhere else
    arm.pick_and_place(
        pick_pos=[0.23, 0.006, 0.015],   # Where the block is
        place_pos=[0.30, 0.10, 0.015],   # Where to drop it
        speed=1.5                         # Slower = safer
    )

    arm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
