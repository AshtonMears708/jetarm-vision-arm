#!/usr/bin/env python3
# coding: utf8
# Custom Color Sorting - Programmable Sequence Logic

import os
import cv2
import yaml
import time
import math
import queue
import rclpy 
import signal
import threading
import numpy as np
from rclpy.node import Node
from sdk import common, fps
from cv_bridge import CvBridge
from interfaces.srv import SetStringBool
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition 
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position
from ros_robot_controller_msgs.msg import MotorsState, MotorState
from servo_controller.action_group_controller import ActionGroupController

PLACE_POSITION = {
    'red': [0.15, -0.08, 0.01],
    'green': [0.15, 0.0, 0.01],
    'blue': [0.15, 0.08, 0.01],
}    

class ColorSortingNode(Node):

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.running = True
        self.start_count = False
        self.lock = threading.RLock()
        self.image_queue = queue.Queue(maxsize=2)
        self.fps = fps.FPS()    
        self.bridge = CvBridge()  
        self.center_imgpts = None
        self.offset = 0.005 
        self.pick_pitch = 80
        self.count = 0
        self.data = common.get_yaml_data("/home/ubuntu/ros2_ws/src/app/config/lab_config.yaml")  
        self.lab_data = self.data['/**']['ros__parameters'] 
        self.camera_type = os.environ.get('CAMERA_TYPE', 'USB_CAM')
        self.min_area = 500
        self.max_area = 15000 
        
        # --- NEW CUSTOM LOGIC VARIABLES ---
        self.current_search_color = 'red' # The active color the brain wants
        self.locked_center_coord = None   # Will hold coordinates when stable
        self.is_moving = False            # Prevents vision from interrupting arm

        self.motor_pub = self.create_publisher(MotorsState, '/ros_robot_controller/set_motor', 1)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        
        # --- FIXED FOR USB CAMERA ---
        self.image_sub = self.create_subscription(Image, '/usb_cam/image_raw', self.image_callback, 1)
        
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/controller_manager/init_finish', callback_group=timer_cb_group)
        self.client.wait_for_service()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()
        
        set_servo_position(self.joints_pub, 1.0, ((1, 850), (2, 630), (3, 90), (4, 85), (5, 480), (10, 200)))
        
        # Spawn the Background Vision Loop and the Brain Logic independently
        threading.Thread(target=self.vision_loop, daemon=True).start()
        threading.Thread(target=self.brain_logic, daemon=True).start()

    # ==========================================================
    # 1. YOUR CUSTOM LOGIC (The "Brain")
    # ==========================================================
    def brain_logic(self):
        """Write your custom sequence here!"""
        time.sleep(3) # Wait for cameras and kinematics to warm up
        self.get_logger().info("System Ready. Starting Custom Sequence...")
        
        self.running = True
        self.start_count = True
        
        # Start the conveyor belt (if attached)
        self.set_motor(1, -100.0)

        # Define exactly what you want it to pick up and in what order
        sequence = ['red', 'blue']

        for color in sequence:
            self.get_logger().info(f"==> Brain: Now searching for {color.upper()}...")
            self.current_search_color = color
            self.locked_center_coord = None
            
            # Wait patiently until the vision thread finds it and locks on
            while self.locked_center_coord is None:
                time.sleep(0.1)
                
            self.get_logger().info(f"==> Brain: Target locked! Executing pickup for {color}.")
            
            # Tell the vision system to pause while the arm moves
            self.is_moving = True 
            
            # Execute the manufacturer's exact movement sequence
            self.move(self.locked_center_coord) 
            
            self.is_moving = False
            self.get_logger().info(f"==> Brain: {color.upper()} sorted successfully.")
            time.sleep(1.0)
            
        self.get_logger().info("Sequence Complete! Stopping motors.")
        self.set_motor(1, 0.0)

    # ==========================================================
    # 2. THE VISION LOOP (Background Task)
    # ==========================================================
    def vision_loop(self):
        """Constantly processes images and updates coordinates"""
        while True:
            target_list, result_image = self.image_processing()
            
            # Only process logic if the arm isn't currently moving
            if target_list and self.running and not self.is_moving:
                
                # We only care about the color the Brain told us to look for!
                if target_list[0][0] == self.current_search_color:
                    self.count += 1
                else:
                    self.count = 0
                    
                # If stable for 65 frames, lock the coordinate so the Brain can trigger move()
                if self.count > 65:
                    self.locked_center_coord = target_list[0][2][0] 
                    self.count = 0 
            else:
                self.count = 0

            # Draw the video window
            if result_image is not None:
                self.fps.update()
                
                # Add text to show what the robot is currently looking for
                cv2.putText(result_image, f"Target: {self.current_search_color}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                cv2.imshow('JetArm Vision', result_image)
                key = cv2.waitKey(1)
                if key == 115: # 's' key to emergency stop
                    self.running = False
                    self.set_motor(1, 0.0)
                    self.get_logger().info("EMERGENCY STOP PRESSED")

    # ==========================================================
    # 3. MANUFACTURER MATH & HARDWARE API (Unchanged)
    # ==========================================================
    def set_motor(self,id, rps):
        data = MotorState()
        data.id = id
        data.rps = rps
        msg = MotorsState()
        msg.data.append(data)
        self.motor_pub.publish(msg)

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def adaptive_threshold(self, gray_image):
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)
        return binary
    
    def canny_proc(self, bgr_image):
        mask = cv2.Canny(bgr_image, 9, 41, 9, L2gradient=True)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))  
        return mask

    def get_top_surface(self, rgb_image):
        image_scale = cv2.convertScaleAbs(rgb_image, alpha=2.5, beta=0)
        image_gray = cv2.cvtColor(image_scale, cv2.COLOR_RGB2GRAY)
        image_mb = cv2.medianBlur(image_gray, 3)  
        image_gs = cv2.GaussianBlur(image_mb, (5, 5), 5)  
        binary = self.adaptive_threshold(image_gs)  
        mask = self.canny_proc(image_gs)  
        mask1 = cv2.bitwise_and(binary, mask)  
        roi_image_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=mask1)  
        return roi_image_mask
 
    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)
 
    def image_processing(self):
        rgb_image = self.image_queue.get()
        result_image = np.copy(rgb_image)
        target_list = []
        index = 0
        if self.start_count :
            img_h, img_w = rgb_image.shape[:2]

            rgb_image = self.get_top_surface(rgb_image)
            image_lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB) 
            
            # --- MODIFIED: ONLY SCAN FOR THE ACTIVE TARGET COLOR ---
            for i in [self.current_search_color]:

                mask = cv2.inRange(image_lab, tuple(self.lab_data['color_range_list'][i]['min']), tuple(self.lab_data['color_range_list'][i]['max']))  
                
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  
                contours_area = map(lambda c: (math.fabs(cv2.contourArea(c)), c), contours)  
                contours = map(lambda a_c: a_c[1], filter(lambda a: self.min_area <= a[0] <= self.max_area, contours_area))
                for c in contours:
                    area = math.fabs(cv2.contourArea(c))
                    rect = cv2.minAreaRect(c)  
                    (center_x, center_y), _ = cv2.minEnclosingCircle(c)
                    center_x, center_y =  center_x,  center_y
                    cv2.circle(result_image, (int(center_x), int(center_y)), 8, (0, 0, 0), -1)
                    corners = list(map(lambda p: (p[0],  p[1]), cv2.boxPoints(rect))) 
                    cv2.drawContours(result_image, [np.intp(corners)], -1, (0, 255, 255), 2, cv2.LINE_AA)  
                    index += 1 
                    angle = int(round(rect[2]))  
                    target_list.append([i, index, (center_x, center_y), angle])
                    
        return target_list, result_image

    def move(self, center_y):
        if self.running:
            y =  ((center_y - 235) / (335 -235)) * (0.11 - 0.16) + 0.16 
            y = y + self.offset
            position = [0, y, 0.08]

            if self.running:
                position[2] += 0.02
                msg = set_pose_target(position, self.pick_pitch,  [-180.0, 180.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse  
                set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), ))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)
                
            if self.running:
                position[2] -= 0.03
                msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse

                set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)

                set_servo_position(self.joints_pub, 1.0, ((10, 600),))
                time.sleep(1)
                
            if self.running:
                position[2] += 0.04
                msg = set_pose_target(position, self.pick_pitch,  [-180.0, 180.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse  

                set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)

                set_servo_position(self.joints_pub, 1.0, ((1, 850), (2, 630), (3, 90), (4, 85), (5, 480)))
                time.sleep(1)

                position =  PLACE_POSITION[self.current_search_color] 

            if self.running:
                position[2] += 0.03
                msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                    
                servo_data = res.pulse  
                set_servo_position(self.joints_pub, 2.0, ((1, servo_data[0]),))
                time.sleep(2)
                set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)

            if self.running:
                position[2] -= 0.03
                msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
                res = self.send_request(self.kinematics_client, msg)
                servo_data = res.pulse 

                set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((10, 200),))
                time.sleep(1)

            if self.running:
                set_servo_position(self.joints_pub, 1.0, ( (2, 630), (3, 90), (4, 85), (5, 480), (10, 200)))
                time.sleep(1)

                set_servo_position(self.joints_pub, 2.0, ((1, 850),))  
                time.sleep(2)

def main():
    node = ColorSortingNode('color_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()