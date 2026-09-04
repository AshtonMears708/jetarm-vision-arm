"""
CV Pick and Place Pipeline
Based directly on manufacturer's positioning_clamp.py

What this does:
    1. Starts camera and color detection
    2. Detects object position automatically
    3. Converts pixel position to real world coordinates
    4. Moves arm to pick up the object
    5. Places it at your defined drop position

Prerequisites:
    Terminal 1: ros2 launch sdk jetarm_sdk.launch.py
    Terminal 2: ros2 launch example color_recognition.launch.py
    Terminal 3: python3 cv_pipeline.py

Customize:
    - PLACE_POSITION: where to drop the object
    - PICK_PITCH: angle of the end effector when picking
    - STABILITY_FRAMES: how many frames before picking
"""

import os
import cv2
import yaml
import time
import rclpy
import threading
import numpy as np
from sdk import common
from rclpy.node import Node
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from app.utils import distortion_inverse_map
from servo_controller import bus_servo_control
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image as RosImage, CameraInfo
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import SetRobotPose
from servo_controller_msgs.msg import ServosPosition


# ── Customize these ───────────────────────────────────────────────────────────

# Where to drop the object after picking (x, y, z in metres)
PLACE_POSITION = [0.30, 0.10, 0.015]

# Gripper
GRIPPER_ID    = 10
GRIPPER_OPEN  = 200
GRIPPER_CLOSE = 550

# Arm settings
PICK_PITCH       = 80     # end effector angle when picking
MOVEMENT_SPEED   = 1.5    # seconds — higher = slower/safer
STABILITY_FRAMES = 60     # frames object must be stable before picking
STABILITY_LIMIT  = 0.01   # metres — max movement to be considered stable

# Safe home position — from manufacturer's code
HOME = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 550))

# Calibration files — same paths as manufacturer
CONFIG_PATH      = '/home/ubuntu/ros2_ws/src/app/config/'
TRANSFORM_FILE   = 'transform.yaml'
CALIBRATION_FILE = 'calibration.yaml'

# ─────────────────────────────────────────────────────────────────────────────


class CVPipeline(Node):
    def __init__(self):
        rclpy.init()
        super().__init__('cv_pipeline',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.bridge        = CvBridge()
        self.intrinsic     = None
        self.distortion    = None
        self.result_image  = None
        self.previous_pose = None
        self.count         = 0
        self.camera_type   = os.environ['CAMERA_TYPE']

        # Servo publisher — correct topic from manufacturer's code
        self.servos_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        # Subscribe to color detection result — already filtered by color
        self.image_sub = self.create_subscription(
            RosImage,
            '/color_detection/result_image',
            self.image_callback,
            1
        )

        # Subscribe to camera info — needed for coordinate conversion
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/depth_cam/rgb/camera_info',
            self.camera_info_callback,
            1
        )

        # Wait for SDK and kinematics to be ready
        self.get_logger().info('Waiting for services...')
        init = self.create_client(Trigger, '/controller_manager/init_finish')
        init.wait_for_service()
        kin = self.create_client(Trigger, '/kinematics/init_finish')
        kin.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()
        self.get_logger().info('All services ready. Looking for object...')

        # Go home first
        self.go_home()

        # Start detection loop in background thread
        threading.Thread(target=self.detection_loop, daemon=True).start()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def image_callback(self, msg):
        """Receives the pre-filtered color detection image."""
        self.result_image = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def camera_info_callback(self, msg):
        """Receives camera calibration data for coordinate conversion."""
        self.intrinsic  = np.matrix(msg.k).reshape(1, -1, 3)
        self.distortion = np.array(msg.d)

    # ── Servo helpers ─────────────────────────────────────────────────────────

    def set_servos(self, positions, speed=None):
        speed = speed or MOVEMENT_SPEED
        bus_servo_control.set_servo_position(self.servos_pub, speed, positions)

    def open_gripper(self):
        self.set_servos(((GRIPPER_ID, GRIPPER_OPEN),))
        time.sleep(1.0)

    def close_gripper(self):
        self.set_servos(((GRIPPER_ID, GRIPPER_CLOSE),))
        time.sleep(1.5)

    def go_home(self):
        self.get_logger().info('Moving to home position...')
        self.set_servos(HOME)
        time.sleep(MOVEMENT_SPEED + 0.5)

    # ── IK movement ───────────────────────────────────────────────────────────

    def send_request(self, msg):
        """Send IK request and wait for response — from manufacturer's code."""
        future = self.kinematics_client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def move_to(self, position):
        """Move end effector to [x, y, z] position using IK."""
        msg = set_pose_target(position, PICK_PITCH, [-90.0, 90.0], 1.0)
        res = self.send_request(msg)
        if res and res.pulse:
            self.set_servos(
                ((1, res.pulse[0]), (2, res.pulse[1]), (3, res.pulse[2]),
                 (4, res.pulse[3]), (5, res.pulse[4]))
            )
            time.sleep(MOVEMENT_SPEED + 0.5)
            return True
        else:
            self.get_logger().error(f'Position {position} is unreachable!')
            return False

    # ── Coordinate conversion — directly from manufacturer's code ─────────────

    def pixel_to_world(self, x, y):
        """Convert pixel coordinates to real world [x, y, z] in metres."""
        with open(CONFIG_PATH + TRANSFORM_FILE, 'r') as f:
            config = yaml.safe_load(f)
            extristric       = np.array(config['extristric'])
            self.white_area_center = np.array(config['white_area_pose_world'])

        tvec = extristric[:1]
        rmat = extristric[1:]
        tvec, rmat = common.extristric_plane_shift(
            np.array(tvec).reshape((3, 1)), np.array(rmat), 0.03
        )

        if self.camera_type == 'USB_CAM':
            x, y = distortion_inverse_map.undistorted_to_distorted_pixel(
                x, y, self.intrinsic, self.distortion
            )

        projection_matrix = np.row_stack((
            np.column_stack((rmat, tvec)),
            np.array([[0, 0, 0, 1]])
        ))

        world_pose = common.pixels_to_world([[x, y]], self.intrinsic, projection_matrix)[0]
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]
        position = self.white_area_center[:3, 3] + world_pose
        world_pose[2] = 0.03

        # Apply pixel calibration offsets
        config_data = common.get_yaml_data(os.path.join(CONFIG_PATH, CALIBRATION_FILE))
        offset = tuple(config_data['pixel']['offset'])
        scale  = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i] + offset[i]
        position[2] = 0.015

        return position

    def apply_kinematics_calibration(self, position):
        """Apply kinematics calibration offsets before moving."""
        config_data = common.get_yaml_data(os.path.join(CONFIG_PATH, CALIBRATION_FILE))
        offset = tuple(config_data['kinematics']['offset'])
        scale  = tuple(config_data['kinematics']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i] + offset[i]
        return position

    # ── Pick and place ────────────────────────────────────────────────────────

    def pick(self, position):
        """Full pick sequence at the given world position."""
        self.get_logger().info(f'Picking at {list(position)}')

        self.open_gripper()

        # Move above object
        above = list(position)
        above[2] = position[2] + 0.05
        if not self.move_to(above):
            return False

        # Move down to object
        if not self.move_to(list(position)):
            return False

        # Grab
        self.close_gripper()

        # Lift up
        self.move_to(above)
        return True

    def place(self, position):
        """Full place sequence at the given world position."""
        self.get_logger().info(f'Placing at {position}')

        above = list(position)
        above[2] = position[2] + 0.05

        self.move_to(above)
        self.move_to(list(position))
        self.open_gripper()
        self.go_home()

    # ── Detection loop ────────────────────────────────────────────────────────

    def detection_loop(self):
        """
        Continuously checks camera for objects.
        When a stable object position is found, picks it up and places it.
        """
        while True:
            try:
                if self.result_image is not None and self.intrinsic is not None:

                    # Find contours in the color-filtered image
                    contours = cv2.findContours(
                        self.result_image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE
                    )[-2]

                    if contours:
                        # Get largest contour
                        c    = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(c)
                        x, y = rect[0][0], rect[0][1]

                        # Convert pixel to world coordinates
                        position = self.pixel_to_world(x, y)

                        self.get_logger().info(f'Pixel: x={x:.0f}, y={y:.0f} → World: {list(np.round(position, 3))}')

                        # Check stability — wait until object stops moving
                        if self.previous_pose is None:
                            self.previous_pose = position
                            continue

                        movement = np.linalg.norm(np.array(position) - np.array(self.previous_pose))

                        if movement < STABILITY_LIMIT:
                            self.count += 1
                        else:
                            self.count = 0
                            self.previous_pose = position

                        if self.count >= STABILITY_FRAMES:
                            # Object is stable — apply kinematics calibration and pick
                            final_position = self.apply_kinematics_calibration(position)
                            self.get_logger().info(f'Object stable! Picking at {list(np.round(final_position, 3))}')

                            if self.pick(final_position):
                                self.place(PLACE_POSITION)

                            # Reset for next object
                            self.count = 0
                            self.previous_pose = None
                            self.get_logger().info('Done. Looking for next object...')

                    else:
                        self.get_logger().info('No object detected — place object in camera view.')
                        self.count = 0
                        self.previous_pose = None

            except Exception as e:
                self.get_logger().error(f'Error: {e}')

            time.sleep(0.05)  # ~20fps detection rate


def main():
    node = CVPipeline()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == '__main__':
    main()
