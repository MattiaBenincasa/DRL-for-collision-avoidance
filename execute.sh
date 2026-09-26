export LIBGL_ALWAYS_SOFTWARE=1
export SVGA_VGPU10=0
export LC_NUMERIC="en_US.UTF-8"
killall -9 gzserver gzclient 2>/dev/null

source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch simulation_layer training.launch
