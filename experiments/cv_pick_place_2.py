"""
CV Pick and Place - Clean Version
Detects objects via camera and moves arm to pick them up.

Prerequisites:
    Terminal 1: ros2 launch sdk jetarm_sdk.launch.py
    Terminal 2: ros2 launch peripherals usb_cam.launch.py
    Terminal 3: ros2 launch example color_recognition.launch.py
    Terminal 4: python3 cv_pick_place.py

Customize:
    PLACE_POSITION: where to drop the object (x, y, z in metres)
    STABILITY_FRAMES: how many frames before picking (higher = more stable)
    MOVEMENT_SPEED: seconds per move (higher = slower/safer)
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


# ── Settings — edit these ─────────────────────────────────────────────────────

PLACE_POSITION   = [0.30, 0.1, 0.07]   # where to drop the object
MOVEMENT_SPEED   = 0.5                    # seconds per move
STABILITY_FRAMES = 60                     # frames before picking
STABILITY_LIMIT  = 0.01                   # metres of allowed movement

GRIPPER_ID    = 10
GRIPPER_OPEN  = 200
GRIPPER_CLOSE = 550
PICK_PITCH    = 80

HOME = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 550))

CONFIG_PATH      = '/home/ubuntu/ros2_ws/src/app/config/'
TRANSFORM_FILE   = 'transform.yaml'
CALIBRATION_FILE = 'calibration.yaml'

# ─────────────────────────────────────────────────────────────────────────────


class CVPickPlace(Node):
    def __init__(self):
        rclpy.init()
        super().__init__('cv_pick_place',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.bridge        = CvBridge()
        self.intrinsic     = None
        self.distortion    = None
        self.result_image  = None
        self.previous_pose = None
        self.count         = 0
        self.camera_type   = os.environ.get('CAMERA_TYPE', 'USB_CAM')

        # Publisher
        self.servos_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        # Camera subscriptions
        self.create_subscription(RosImage, '/color_detection/result_image', self.image_callback, 1)
        self.create_subscription(CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1)

        # Wait for services
        self.get_logger().info('Waiting for services...')
        c1 = self.create_client(Trigger, '/controller_manager/init_finish')
        c1.wait_for_service()
        c2 = self.create_client(Trigger, '/kinematics/init_finish')
        c2.wait_for_service()
        self.ik_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.ik_client.wait_for_service()
        self.get_logger().info('Ready. Place object in camera view...')

        self.go_home()
        threading.Thread(target=self.detection_loop, daemon=True).start()

    # ── Camera callbacks ──────────────────────────────────────────────────────

    def image_callback(self, msg):
        self.result_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')

    def camera_info_callback(self, msg):
        self.intrinsic  = np.matrix(msg.k).reshape(1, -1, 3)
        self.distortion = np.array(msg.d)

    # ── Servo control ─────────────────────────────────────────────────────────

    def set_servos(self, positions, speed=None):
        bus_servo_control.set_servo_position(self.servos_pub, speed or MOVEMENT_SPEED, positions)

    def open_gripper(self):
        self.set_servos(((GRIPPER_ID, GRIPPER_OPEN),))
        time.sleep(1.0)

    def close_gripper(self):
        self.set_servos(((GRIPPER_ID, GRIPPER_CLOSE),))
        time.sleep(1.5)

    def go_home(self):
        self.get_logger().info('Going home...')
        self.set_servos(HOME)
        time.sleep(MOVEMENT_SPEED + 0.5)

    # ── IK movement ───────────────────────────────────────────────────────────

    def move_to(self, position):
        """Move to [x, y, z] using IK. Returns True if successful."""
        msg = set_pose_target(position, PICK_PITCH, [-90.0, 90.0], 1.0)
        future = self.ik_client.call_async(msg)
        timeout = time.time() + 5.0
        # Use spin_until_future_complete — same pattern as working manual_control.py

	
        while rclpy.ok() and time.time() < timeout:
            if future.done() and future.result():
                res = future.result()
                if res.pulse:
                    self.get_logger().info(f'Moving to {list(np.round(position, 3))}')
                    self.set_servos((
                        (1, res.pulse[0]),
                        (2, res.pulse[1]),
                        (3, res.pulse[2]),
                        (4, res.pulse[3]),
                        (5, res.pulse[4])
                    ))
                    time.sleep(MOVEMENT_SPEED + 0.5)
                    return True

        self.get_logger().error(f'Could not reach {position}')
        return False

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def pixel_to_world(self, x, y):
        with open(CONFIG_PATH + TRANSFORM_FILE, 'r') as f:
            config = yaml.safe_load(f)
            extristric = np.array(config['extristric'])
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

        proj = np.row_stack((
            np.column_stack((rmat, tvec)),
            np.array([[0, 0, 0, 1]])
        ))

        world_pose = common.pixels_to_world([[x, y]], self.intrinsic, proj)[0]
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]
        position = self.white_area_center[:3, 3] + world_pose

        config_data = common.get_yaml_data(os.path.join(CONFIG_PATH, CALIBRATION_FILE))
        offset = tuple(config_data['pixel']['offset'])
        scale  = tuple(config_data['pixel']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i] + offset[i]
        position[2] = 0.015

        return position

    def apply_calibration(self, position):
        config_data = common.get_yaml_data(os.path.join(CONFIG_PATH, CALIBRATION_FILE))
        offset = tuple(config_data['kinematics']['offset'])
        scale  = tuple(config_data['kinematics']['scale'])
        for i in range(3):
            position[i] = position[i] * scale[i] + offset[i]
        return position

    # ── Pick and place ────────────────────────────────────────────────────────

    def pick(self, position):
        self.get_logger().info(f'Picking at {list(np.round(position, 3))}')
        above = list(position)
        above[2] = float(position[2]) + 0.05

        self.open_gripper()
        if not self.move_to(above):
            return False
        if not self.move_to(list(position)):
            return False
        self.close_gripper()
        self.move_to(above)
        return True

    def place(self, position):
        self.get_logger().info(f'Placing at {position}')
        above = list(position)
        above[2] = position[2] + 0.05
        self.move_to(above)
        self.move_to(list(position))
        self.open_gripper()
        self.go_home()

    # ── Detection loop ────────────────────────────────────────────────────────

    def detection_loop(self):
        while True:
            try:
                if self.result_image is not None and self.intrinsic is not None:
                    contours = cv2.findContours(
                        self.result_image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE
                    )[-2]

                    if contours:
                        c    = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(c)
                        x, y = rect[0][0], rect[0][1]

                        position = self.pixel_to_world(x, y)
                        self.get_logger().info(f'Pixel ({x:.0f},{y:.0f}) → World {list(np.round(position, 3))}')

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
                            final = self.apply_calibration(position)
                            self.get_logger().info(f'Stable! Picking at {list(np.round(final, 3))}')
                            if self.pick(final):
                                self.place(PLACE_POSITION)
                            self.count = 0
                            self.previous_pose = None
                            self.get_logger().info('Done. Looking for next object...')
                    else:
                        self.get_logger().info('No object detected.')
                        self.count = 0
                        self.previous_pose = None

            except Exception as e:
                self.get_logger().error(f'Error: {e}')

            time.sleep(0.05)


def main():
    node = CVPickPlace()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == '__main__':
    main()
