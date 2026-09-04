"""
Controlled Color Sorting — Based on manufacturer's cv_control.py
Uses pixel_to_world coordinate conversion from cv_pick_place2.py (the version that worked)

Structure:
    - classify_utensil()    — runs YOLOv11n to identify Utensil  shown to camera
    - decide_pick_order() — filters detected blocks by the utensil-driven goal color
    - PLACE_POSITIONS     — define where each color gets dropped
    - TARGET_COLORS       — which colors to look for
    - UTENSIL_TO_COLOR      — maps utensil class name to pick color goal
    - UTENSIL_SCAN_HOME     — servo positions for horizontal utensil-scanning pose

Prerequisites:
    Terminal 1: ros2 launch sdk jetarm_sdk.launch.py
    Terminal 2: ros2 launch peripherals usb_cam.launch.py
    Terminal 3: python3 color_sort_controlled.py
    Press 'A' in camera window to start, 'S' to stop
"""

import os
import cv2
import yaml
import math
import time
import queue
import rclpy
import threading
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
from sdk import common, fps
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from app.utils import distortion_inverse_map
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from servo_controller_msgs.msg import ServosPosition
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import SetRobotPose
from servo_controller.bus_servo_control import set_servo_position


# ── Customize these ───────────────────────────────────────────────────────────

# Where to drop each color (x, y, z in metres)
PLACE_POSITIONS = {
    'red':   [0.15, -0.16, 0.05],
    'green': [0.15,  0.00, 0.05],
    'blue':  [0.15,  0.16, 0.05],
}

# Which colors to detect
TARGET_COLORS = ['red', 'green', 'blue']

# Arm settings
PICK_PITCH   = 80
MIN_AREA     = 500
MAX_AREA     = 15000
STABLE_COUNT = 65

#DEBUG
debugBoolean = True

# Home position — arm looks down at table for object detection
HOME = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 500))
HOME_OPEN = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 500), (10, 200))

# Transit pose — raised arm used when carrying a block between pick and place.
# Servo 5 is intentionally OMITTED so the wrist roll holding the block is preserved.
# TUNE servos 2/3/4 so the gripper clears the workspace; values below are a starting point.
TRANSIT_POSE = ((1, 500), (2, 500), (3, 250), (4, 200))

# Utensil scan position — arm looks horizontally to classify Utensil shown to camera
# Update these servo values after physically positioning the arm
UTENSIL_SCAN_HOME = ((1, 500), (2, 520), (3, 210), (4, 200), (5, 500))

# Servo calibration for orientation / wrist-roll control — TUNE for your hardware
SERVO1_PULSE_PER_DEGREE = 5.0     # Base servo: pulses per degree (existing pick code uses 5)
SERVO5_NEUTRAL_PULSE    = 500     # Servo 5 pulse value when gripper roll is at 0 degrees
SERVO5_PULSE_PER_DEGREE = 5.0 # Gripper roll: pulses per degree
BASE_COMP_SIGN          = 1.0     # Set to -1.0 if base-rotation compensation rotates the wrong way

# IK pitch tolerance — kept consistent across all phases so IK returns the same arm
# configuration at approach, descent, lift, and place (prevents elbow/wrist flips between phases)
IK_PITCH_TOL = [-90.0, 90.0]

# Frames the same utensil must be detected consecutively before locking in the goal
UTENSIL_STABLE_COUNT = 20

# Mapping from YOLO utensil class name to the pick color goal
UTENSIL_TO_COLOR = {
    'fork':  'red',
    'knife': 'green',
    'spoon': 'blue',
}

# Calibration files
CONFIG_PATH      = '/home/ubuntu/ros2_ws/src/app/config/'
TRANSFORM_FILE   = 'transform.yaml'
CALIBRATION_FILE = 'calibration.yaml'

#Min and Max reach of X
MIN_REACH = 0.05
MAX_REACH = 0.28

# ─────────────────────────────────────────────────────────────────────────────



def decide_pick_order(detected_blocks, goal_color):

    if not detected_blocks or goal_color is None:
        return None

    matches = [b for b in detected_blocks if b[0] == goal_color]

    if not matches:
        return None

    # Pick the largest contour area as the most confident detection
    return max(matches, key=lambda b: b[3] * b[4])

# ─────────────────────────────────────────────────────────────────────────────


class ControlledColorSort(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name,
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.running     = False
        self.start_count = False
        self.lock        = threading.RLock()
        self.image_queue = queue.Queue(maxsize=2)
        self.fps_counter = fps.FPS()
        self.bridge      = CvBridge()
        self.target      = None
        self.count       = 0
        self.pick_pitch  = PICK_PITCH
        self.intrinsic   = None
        self.distortion  = None
        self.camera_type = os.environ.get('CAMERA_TYPE', 'USB_CAM')

        # Mode state — the system alternates between scanning for utensil and picking objects
        # 'utensil'  — arm at UTENSIL_SCAN_HOME, YOLO classifying utensil shown to camera
        # 'pick'   — arm looking down at table, OpenCV finding matching colored object
        self.mode         = 'utensil'
        self.goal_color   = None
        self.utensil_count  = 0

        # Load the YOLO model for utensil classification
        self.utensil_model = YOLO('/home/ubuntu/third_party/yolo/yolo11n.pt')
        if debugBoolean:
            self.get_logger().info('utensil YOLO model loaded.')

        # Load color range config
        self.lab_data = {
        	'color_range_list': {
        		'red': {
        			'min': [0, 146, 146],
        			'max': [255, 187, 207],
        		},
        		'green': {
        			'min': [0, 49, 100],
        			'max': [255, 113, 197],
        		},
        		'blue': {
        			'min': [0, 115, 30],
        			'max': [255, 220, 100],
        		},
        	}
        }

        # Publishers
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)

        # Camera subscriptions
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 1
        )
        self.create_subscription(
            CameraInfo, '/depth_cam/rgb/camera_info', self.camera_info_callback, 1
        )

        # Services
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(
            Trigger, '/controller_manager/init_finish', callback_group=timer_cb_group
        )
        self.client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()
        
        if debugBoolean:
            self.get_logger().info('waiting for camera info...')
        while self.intrinsic is None:
            rclpy.spin_once(self, timeout_sec = 0.1)
        if debugBoolean:
            self.get_logger().info('Ready. Press A in the camera window to start, S to stop.')

        # Move to utensil scan position — arm looks horizontally at startup
        set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
        time.sleep(1.5)

        threading.Thread(target=self.main_loop, daemon=True).start()

    # ── Camera callbacks ──────────────────────────────────────────────────────

    def image_callback(self, ros_image):
        cv_image  = self.bridge.imgmsg_to_cv2(ros_image, 'rgb8')
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def camera_info_callback(self, msg):
        self.intrinsic  = np.matrix(msg.k).reshape(1, -1, 3)
        self.distortion = np.array(msg.d)

    # ── Vision helpers ────────────────────────────────────────────────────────

    def adaptive_threshold(self, gray):
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10
        )

    def canny_proc(self, bgr):
        mask = cv2.Canny(bgr, 9, 41, 9, L2gradient=True)
        return 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    def get_top_surface(self, rgb):
        scaled  = cv2.convertScaleAbs(rgb, alpha=4, beta=0.5)
        gray    = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        blurred = cv2.GaussianBlur(blurred, (5, 5), 5)
        binary  = self.adaptive_threshold(blurred)
        edges   = self.canny_proc(blurred)
        mask    = cv2.bitwise_and(binary, edges)
        return cv2.bitwise_and(rgb, rgb, mask=mask)

    # ── Coordinate conversion — from cv_pick_place2.py (the one that worked) ──

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

    # ── Image processing ──────────────────────────────────────────────────────

    def image_processing(self):
        rgb_image    = self.image_queue.get()
        result_image = np.copy(rgb_image)
        target_list  = []

        if self.start_count:
            processed = self.get_top_surface(rgb_image)
            image_lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)

            for color in TARGET_COLORS:
                mask = cv2.inRange(
                    image_lab,
                    tuple(self.lab_data['color_range_list'][color]['min']),
                    tuple(self.lab_data['color_range_list'][color]['max'])
                )
                eroded   = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated  = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

                for c in contours:
                    area = math.fabs(cv2.contourArea(c))
                    if not (MIN_AREA <= area <= MAX_AREA):
                        continue

                    rect = cv2.minAreaRect(c)
                    (cx, cy), _ = cv2.minEnclosingCircle(c)
                    (rect_cx, rect_cy), (width, height), angle = rect
                    angle = int(round(angle))

                    cv2.circle(result_image, (int(cx), int(cy)), 8, (0, 0, 0), -1)
                    corners = list(map(lambda p: (p[0], p[1]), cv2.boxPoints(rect)))
                    cv2.drawContours(result_image, [np.intp(corners)], -1, (0, 255, 255), 2)
                    cv2.putText(result_image, color, (int(cx) + 10, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    target_list.append([color, (cx, cy), angle, width, height])

        # Return raw rgb_image alongside result so YOLO always gets a clean unannotated frame
        return target_list, result_image, rgb_image

    # ── IK helper — exact from manufacturer ──────────────────────────────────

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    # ── Pick and place ────────────────────────────────────────────────────────

    def pick_and_place(self, block):

        color, (center_x, center_y), angle = block[0], block[1], block[2]
        if debugBoolean:
            self.get_logger().info(f'Picking {color} at pixel ({center_x:.0f}, {center_y:.0f})')

        # Convert pixel to world — same as cv_pick_place2.py
        position = self.pixel_to_world(center_x, center_y)
        position = self.apply_calibration(position)
        position = list(position)
        if debugBoolean:
            self.get_logger().info(f'World: {[round(p, 3) for p in position]}')
        
        x, y, z = position
        #Tuning needed
        # if not (0.05 <= x <= 0.28) or not (-0.15 <= y <= 0.15):
        # Change this if something breaks with boundaries 
        distance = math.sqrt(x**2 + y**2)
        if not (MIN_REACH <= distance <= MAX_REACH):
            if debugBoolean:
                self.get_logger().warn(f'Block out of range at {[round(p,3) for p in position]} - skipping')
            self.count       = 0
            self.target      = None
            self.goal_color  = None
            self.utensil_count = 0
            self.mode        = 'utensil'
            self.start_count = False
            set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
            time.sleep(1.5)
            return

        # Approach above
        position[2] += 0.05
        msg = set_pose_target(position, self.pick_pitch, IK_PITCH_TOL, 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if not res or not res.pulse:
            if debugBoolean:
                self.get_logger().error('IK failed on approach')
            self.count       = 0
            self.target      = None
            self.goal_color  = None
            self.utensil_count = 0
            self.mode        = 'utensil'
            self.start_count = False
            set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
            time.sleep(1.5)
            return
        servo_data = res.pulse
        set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
        time.sleep(1)
        set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
        time.sleep(1)

        # Get dimensions to find long axis (gripper jaws align WITH long axis so they
        # close ACROSS the short axis of the block)
        color, (center_x, center_y), angle, width, height = block

        # Open gripper
        set_servo_position(self.joints_pub, 0.5, ((10, 200),))
        time.sleep(1)

        # ── Compute servo 5 (gripper roll) ─────────────────────────────────────
        if width > height:
            angle -= 90 # Tuning logic placeholder

        servo_5_pos = int(550 + (angle / 90) * 450)
        servo_5_pos = max(0, min(1000, servo_5_pos))

        # Command just the gripper rotation servo (Servo 5)
        set_servo_position(self.joints_pub, 0.3, ((5, servo_5_pos),))
            
        if debugBoolean:
            self.get_logger().info(
                f'Width: {width:.1f} Height: {height:.1f} block_angle: {angle:.1f} '

                f'servo5: {servo_5_pos}'
            )
        time.sleep(0.5)

        # Lower to pick
        position[2] -= 0.05
        msg = set_pose_target(position, self.pick_pitch, IK_PITCH_TOL, 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if not res or not res.pulse:
            if debugBoolean:
                self.get_logger().error('IK failed on lower')
            self.count       = 0
            self.target      = None
            self.goal_color  = None
            self.utensil_count = 0
            self.mode        = 'utensil'
            self.start_count = False
            set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
            time.sleep(1.5)
            return
        servo_data = res.pulse
        set_servo_position(self.joints_pub, 0.5, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
        time.sleep(1)

       

        # Close gripper
        set_servo_position(self.joints_pub, 1, ((10, 600),))
        time.sleep(1)
        

        # Lift up
        position[2] += 0.05
        msg = set_pose_target(position, self.pick_pitch, IK_PITCH_TOL, 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

        # Move to transit pose — raised, and crucially does NOT include servo 5,
        # so the gripper roll holding the block is preserved
        set_servo_position(self.joints_pub, 1.0, TRANSIT_POSE)
        time.sleep(1.5)

        # Move to place position
        place_pos = list(PLACE_POSITIONS[color])
        if debugBoolean:
            self.get_logger().info(f'Placing {color} at {place_pos}')

        place_pos[2] += 0.05
        msg = set_pose_target(place_pos, self.pick_pitch, IK_PITCH_TOL, 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
            time.sleep(2)
            # Apply servo 5 from IK so the block is placed at the IK-chosen orientation
            # rather than whatever roll it happened to be picked at
            set_servo_position(
                self.joints_pub, 1.0,
                ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3]), (5, servo_data[4]))
            )
            time.sleep(1)

        # Lower to place
        place_pos[2] -= 0.05
        msg = set_pose_target(place_pos, self.pick_pitch, IK_PITCH_TOL, 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(
                self.joints_pub, 0.5,
                ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3]))
            )
            time.sleep(1)

        # Open gripper
        set_servo_position(self.joints_pub, 0.5, ((10, 200),))
        time.sleep(1)

        # Return home
        set_servo_position(self.joints_pub, 1.0, HOME)
        time.sleep(1.5)

        if debugBoolean:
            self.get_logger().info('Done. Returning to utensil scan mode.')
        self.target      = None
        self.goal_color  = None
        self.utensil_count = 0
        self.mode        = 'utensil'
        self.start_count = False
        set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
        time.sleep(1.5)

    def classify_utensil(self, rgb_image):
        """
        Runs YOLO on the current frame to detect utensil shown to the camera.
        Returns the mapped goal color string if a utensil is detected confidently,
        or None if nothing relevant is seen.
        """
        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        results = self.utensil_model(rgb_image, verbose=False, conf=0.65)

        best_conf  = 0.0
        best_color = None

        for r in results:
            for box in r.boxes:
                label = self.utensil_model.names[int(box.cls)]
                conf  = float(box.conf)
                if label in UTENSIL_TO_COLOR and conf > best_conf:
                    best_conf  = conf
                    best_color = UTENSIL_TO_COLOR[label]
                    if debugBoolean:
                        self.get_logger().info(f'YOLO saw: {label} ({conf:.2f})')

        return best_color

    # ── Main loop ─────────────────────────────────────────────────────────────

    def main_loop(self):
        while True:
            target_list, result_image, raw_image = self.image_processing()

            if self.running:

                if self.mode == 'utensil':
                    # Utensil scanning mode — run YOLO on clean raw frame
                    detected_color = self.classify_utensil(raw_image)

                    if detected_color is not None:
                    
                        if detected_color == self.goal_color:
                            self.utensil_count += 1
                        else:
                            self.goal_color  = detected_color
                            self.utensil_count = 1
                    else:
                        if self.utensil_count <= 0:
                            self.utensil_count = 0
                        else:
                        # PUNISH ERRORS
                            self.utensil_count -= 2
                        
                    if self.utensil_count >= UTENSIL_STABLE_COUNT:
                        if debugBoolean:
                            self.get_logger().info(
                                f'Utensil detected: {self.goal_color} goal locked. Switching to pick mode.'
                            )
                        self.utensil_count = 0
                        self.mode        = 'pick'
                        set_servo_position(self.joints_pub, 1.0, HOME)
                        time.sleep(1.5)
                        self.start_count = True

                elif self.mode == 'pick':
                    chosen = decide_pick_order(target_list, self.goal_color)

                    if chosen is not None: 
                        if self.target is not None and self.target[0] == chosen[0]:
                            self.target = chosen 
                            self.count += 1
                        else:
                            self.target = chosen
                            self.count  = 0
                    else:
                        self.count = 0

                    if self.count > STABLE_COUNT and self.target is not None:
                        self.count       = 0
                        self.start_count = False
                        threading.Thread(
                            target=self.pick_and_place,
                            args=(self.target,),
                            daemon=True
                        ).start()

            if result_image is not None:
                self.fps_counter.update()
                display = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

                cv2.putText(display, f'FPS: {self.fps_counter.fps:.1f}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                status = 'RUNNING — press S to stop' if self.running else 'STOPPED — press A to start'
                cv2.putText(display, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if self.running else (0, 0, 255), 2)

                cv2.putText(display, f'Mode: {self.mode.upper()}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

                if self.goal_color:
                    cv2.putText(display, f'Goal: {self.goal_color}', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                if self.mode == 'utensil':
                    cv2.putText(display,
                                f'Utensil stable: {self.utensil_count}/{UTENSIL_STABLE_COUNT}',
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                if self.mode == 'pick' and self.target:
                    cv2.putText(display,
                                f'Target: {self.target[0]}  ({self.count}/{STABLE_COUNT})',
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                cv2.imshow('Color Sorting — Camera View', display)
                key = cv2.waitKey(1)
                if key == ord('a'):
                    self.running     = True
                    self.mode        = 'utensil'
                    self.goal_color  = None
                    self.utensil_count = 0
                    self.target      = None
                    self.count       = 0
                    set_servo_position(self.joints_pub, 1.0, UTENSIL_SCAN_HOME)
                    time.sleep(1.5)
                    if debugBoolean:
                        self.get_logger().info('Started. Show a utensil to the camera.')
                elif key == ord('s'):
                    self.running     = False
                    self.mode        = 'utensil'
                    self.goal_color  = None
                    self.utensil_count = 0
                    self.target      = None
                    self.count       = 0
                    if debugBoolean:
                        self.get_logger().info('Stopped.')
      


def main():
    node = ControlledColorSort('color_sorting_controlled')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
