#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ros_launch_file> [headless]"
    exit 1
fi

source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

HEADLESS_ARGS=""

if [ "$2" == "headless" ]; then
    HEADLESS_ARGS="gui:=false headless:=true"
fi

roslaunch simulation_layer $1 $HEADLESS_ARGS
