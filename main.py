import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class MazeSolver(Node):
    def __init__(self):
        super().__init__('maze_solver')

        # Publisher for robot velocity
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for LaserScan
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Subscriber for camera feed
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

        # Movement control variables
        self.twist = Twist()
        self.red_detected = False

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Red color range in HSV
        self.red_lower_bound = np.array([0, 120, 70])
        self.red_upper_bound = np.array([10, 255, 255])
        self.red_lower_bound_2 = np.array([170, 120, 70])
        self.red_upper_bound_2 = np.array([180, 255, 255])

    def laser_callback(self, msg):
        if self.red_detected:
            return  # Stop maze navigation if red object is detected

        # Process laser data for obstacle avoidance
        left = min(min(msg.ranges[60:120]), 10.0)  # Left side
        front = min(min(msg.ranges[0:30] + msg.ranges[330:360]), 10.0)  # Front
        right = min(min(msg.ranges[240:300]), 10.0)  # Right side

        safe_distance = 0.5  # Distance to avoid obstacles

        if front < safe_distance:
            # Turn to avoid obstacle
            if left > right:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.5  # Turn left
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = -0.5  # Turn right
        else:
            # Move forward if the path is clear
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0

        # Publish velocity command
        self.cmd_vel_publisher.publish(self.twist)

    def camera_callback(self, msg):
        if self.red_detected:
            # Adjust movement to approach the red object
            self.follow_red_object(msg)
            return

        # Convert image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv_frame, self.red_lower_bound, self.red_upper_bound)
        mask2 = cv2.inRange(hsv_frame, self.red_lower_bound_2, self.red_upper_bound_2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Check if red object is detected
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self.red_detected = True
            self.get_logger().info("Red object detected! Switching to follow mode.")
            self.follow_red_object(msg)

        # Optionally display the camera feed and mask (for debugging)
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)

    def follow_red_object(self, msg):
        # Convert image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv_frame, self.red_lower_bound, self.red_upper_bound)
        mask2 = cv2.inRange(hsv_frame, self.red_lower_bound_2, self.red_upper_bound_2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of red in the frame
        red_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        red_percentage = (red_area / total_area) * 100

        # Stop if the red object dominates the frame
        if red_percentage > 90.0:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(self.twist)
            self.get_logger().info("Got you, red object!")
            return

        # Find the largest contour of the red object
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the red object
            center_x = x + w // 2
            frame_center_x = frame.shape[1] // 2

            # Adjust robot motion to center on the red object
            if abs(center_x - frame_center_x) > 50:
                if center_x < frame_center_x:
                    self.twist.angular.z = 0.3  # Turn left
                else:
                    self.twist.angular.z = -0.3  # Turn right
                self.twist.linear.x = 0.0
            else:
                self.twist.linear.x = 0.2  # Move forward
                self.twist.angular.z = 0.0

        # Publish velocity command
        self.cmd_vel_publisher.publish(self.twist)

        # Optionally display the camera feed and mask (for debugging)
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    maze_solver = MazeSolver()
    rclpy.spin(maze_solver)
    maze_solver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
