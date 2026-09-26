#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan


class LaserSensor:

    def __init__(self):
        # Initialize ROS subscribers
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.scan_data = None

    def laser_callback(self, data):
        arr = np.array(data.ranges)
        arr = np.nan_to_num(arr, nan=5.0, posinf=5.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 5.0)
        self.scan_data = arr