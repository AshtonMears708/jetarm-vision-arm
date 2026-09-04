"""
Controlled Color Sorting — Based on manufacturer's cv_control.py
Uses pixel_to_world coordinate conversion from cv_pick_place2.py (the version that worked)

Structure:
    - decide_pick_order() — THIS is where you plug in your AI logic later
    - PLACE_POSITIONS     — define where each color gets dropped
    - TARGET_COLORS       — which colors to look for

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
import torch
from torchvision import transforms
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

# Home position
HOME = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 550))
HOME_OPEN = ((1, 500), (2, 520), (3, 210), (4, 50), (5, 550), (10, 200))

# Calibration files
CONFIG_PATH      = '/home/ubuntu/ros2_ws/src/app/config/'
TRANSFORM_FILE   = 'transform.yaml'
CALIBRATION_FILE = 'calibration.yaml'

# ─────────────────────────────────────────────────────────────────────────────


# ── PLUG YOUR AI LOGIC IN HERE ────────────────────────────────────────────────
def decide_pick_order(detected_blocks, goal_color=None):
    """
    Given a list of detected blocks and an optional goal color, return which
    one to pick up next.

    detected_blocks: list of [color, (cx, cy), angle, width, height]
    goal_color:      color string set by the fruit classifier, or None
    Returns:         one block or None

    When goal_color is set by the fruit classifier, only blocks matching that
    color are considered. When None, picks the first detected block.
    """
    if not detected_blocks:
        return None

    if goal_color is None:
        return detected_blocks[0]

    for block in detected_blocks:
        if block[0] == goal_color:
            return block

    return None

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

        # Goal color set by the fruit classifier each frame
        self.current_goal_color = None

        # Load fruit classification model from USB drive
        self.fruit_model = torch.jit.load('/media/ubuntu/5934-19DC/fruit_classifier.pt')
        self.fruit_model.eval()
        self.fruit_classes = ['Apple', 'Banana', 'Grape']

        # Image transform matching the preprocessing used during training
        self.fruit_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Publishers
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)

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
	
        self.get_logger().info('waiting for camera info...')
        while self.intrinsic is None:
            rclpy.spin_once(self, timeout_sec = 0.1)
        self.get_logger().info('Ready. Press A in the camera window to start, S to stop.')

        # Go home
        set_servo_position(self.joints_pub, 1.0, HOME)
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

        return target_list, result_image

    # ── IK helper — exact from manufacturer ──────────────────────────────────

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    # ── Pick and place ────────────────────────────────────────────────────────

    def pick_and_place(self, block):

        color, (center_x, center_y), angle = block[0], block[1], block[2]
        self.get_logger().info(f'Picking {color} at pixel ({center_x:.0f}, {center_y:.0f})')

        # Convert pixel to world — same as cv_pick_place2.py
        position = self.pixel_to_world(center_x, center_y)
        position = self.apply_calibration(position)
        position = list(position)
        self.get_logger().info(f'World: {[round(p, 3) for p in position]}')
        
        x, y, z = position
        if not (0.05 <= x <= 0.28) or not (-0.15 <= y <= 0.15):
            self.get_logger().warn(f'Block out of range at {[round(p,3) for p in position]} - skipping')
            self.count = 0
            self.target = None 
            self.start_count = True 
            return 

        # Approach above
        position[2] += 0.05
        msg = set_pose_target(position, self.pick_pitch, [-180.0, 180.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if not res or not res.pulse:
            self.get_logger().error('IK failed on approach')
            self.start_count = True
            return
        servo_data = res.pulse
        set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
        time.sleep(1)
        set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
        time.sleep(1)

        # Open gripper
        set_servo_position(self.joints_pub, 0.5, ((10, 200),))
        time.sleep(1)

        # Lower to pick
        position[2] -= 0.04
        msg = set_pose_target(position, self.pick_pitch, [-90.0, 90.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if not res or not res.pulse:
            self.get_logger().error('IK failed on lower')
            self.start_count = True
            return
        servo_data = res.pulse
        set_servo_position(self.joints_pub, 0.5, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
        time.sleep(1)


        # Get dimensions to find short axis 
        color, (center_x, center_y), angle, width, height = block

        
        if width > height:
            angle = angle + (125 * (height/width))
            
        
        
        # Rotate Gripper
        servo_5_pos = int(550 + (angle / 90) * 450)
        servo_5_pos = max(0, min(1000, servo_5_pos))
        set_servo_position(self.joints_pub, 0.3, ((5, servo_5_pos),))
        self.get_logger().info(f'Width: {width:.1f} Height: {height:.1f} raw_angle: {angle:.1f} servo5: {servo_5_pos}')
        time.sleep(0.5)	

        # Close gripper
        set_servo_position(self.joints_pub, 0.5, ((10, 600),))
        time.sleep(1)
        

        # Lift up
        position[2] += 0.05
        msg = set_pose_target(position, self.pick_pitch, [-180.0, 180.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

        # Return home
        set_servo_position(self.joints_pub, 1.0, HOME)
        time.sleep(1.5)

        # Move to place position
        place_pos = list(PLACE_POSITIONS[color])
        self.get_logger().info(f'Placing {color} at {place_pos}')

        place_pos[2] += 0.05
        msg = set_pose_target(place_pos, self.pick_pitch, [-90.0, 90.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
            time.sleep(2)
            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

        # Lower to place
        place_pos[2] -= 0.05
        msg = set_pose_target(place_pos, self.pick_pitch, [-90.0, 90.0], 1.0)
        res = self.send_request(self.kinematics_client, msg)
        if res and res.pulse:
            servo_data = res.pulse
            set_servo_position(self.joints_pub, 0.5, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

        # Open gripper
        set_servo_position(self.joints_pub, 0.5, ((10, 200),))
        time.sleep(1)

        # Return home
        set_servo_position(self.joints_pub, 1.0, HOME)
        time.sleep(1.5)

        self.get_logger().info('Done. Ready for next block.')
        self.target = None
        self.start_count = True

    def classify_fruit(self, rgb_image):
        """
        Runs the fruit classifier on the current camera frame.
        Maps the predicted fruit to the corresponding pick color goal.
        Returns None if the model is not confident enough to act on.
        """
        tensor = self.fruit_transform(rgb_image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.fruit_model(tensor)
            probs   = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        if conf.item() < 0.80:
            return None

        fruit = self.fruit_classes[pred.item()]

        # Map fruit label to the color of the object the arm should pick up
        fruit_to_color = {
            'Apple':  'red',
            'Banana': 'green',
            'Grape':  'blue',
        }
        return fruit_to_color.get(fruit, None)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def main_loop(self):
        while True:
            target_list, result_image = self.image_processing()

            if target_list and self.running:
                # Run fruit classifier on current frame to set pick goal
                self.current_goal_color = self.classify_fruit(result_image)

                chosen = decide_pick_order(target_list, self.current_goal_color)
                if chosen is not None:
                    if self.target is not None and self.target[0] == chosen[0]:
                        self.count += 1
                    else:
                        self.target = chosen
                        self.count = 0
            else:
                # nothing in frame so reset count
                self.count = 0
                self.previous_pose = None   

            if self.count > STABLE_COUNT and self.target is not None:
                self.count = 0
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

                if self.target:
                    cv2.putText(display,
                                f'Target: {self.target[0]}  ({self.count}/{STABLE_COUNT})',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                cv2.imshow('Color Sorting — Camera View', display)
                key = cv2.waitKey(1)
                if key == ord('a'):
                    self.running     = True
                    self.start_count = True
                    self.get_logger().info('Started sorting.')
                elif key == ord('s'):
                    self.running = False
                    self.count   = 0
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
