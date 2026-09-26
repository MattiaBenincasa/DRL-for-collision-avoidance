#!/usr/bin/env python3

import rospy

def minimal_controller(command_manager):
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        command_manager.step(0);        # w = -0.8 rad/s
        rate.sleep()

        
