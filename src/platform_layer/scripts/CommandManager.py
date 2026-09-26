#!/usr/bin/env python3

import rospy
import numpy as np
import time
from geometry_msgs.msg import Twist


class CommandManager:
    
    def __init__(self, status_reporter):
        # TODO change queue_size = 1 
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.status_reporter = status_reporter

    def step(self, action):
        # action in [0..10]

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.3

        omega = -0.8 + 0.16 * action
        vel_cmd.angular.z = float(omega)
        self.pub_cmd_vel.publish(vel_cmd)