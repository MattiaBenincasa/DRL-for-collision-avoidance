#!/usr/bin/env python3
"""
run_DWA.py — Entry point for the DWA local planner.

Follows the same pattern as training_paper.py: the execution logic
is inline in the script, not encapsulated in a `run()` method of the class.

Usage:
  roslaunch simulation_layer DWA.launch
  rosrun simulation_layer run_DWA.py
"""

import os
import sys
import math

# Add workspace src to PYTHONPATH so cross-package imports work
_repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

import rospy
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from control_layer.scripts.DWA import DWA

rospy.init_node('dwa_planner', anonymous=False)

node = DWA()
rate = rospy.Rate(10)

# ── Phase 1: Spawn ────────────────────────────────────────
rospy.loginfo('DWA: spawning robot ...')
node._spawn_robot()

# ── Phase 2: Wait for sensor data ─────────────────────────
rospy.wait_for_message('/scan', LaserScan)
rospy.wait_for_message('/gazebo/model_states', ModelStates)

# ── Phase 3: Move robot to ParseWorld spawn ───────────────
node._respawn_at_spawn()

rospy.loginfo('DWA: got initial data. Starting control loop.')

# Cache SetModelState proxy for velocity commands
_set_state_proxy = None

# ── Phase 4: Control loop ─────────────────────────────────
while not rospy.is_shutdown():
    if not node.pose_ready:
        rate.sleep()
        continue

    # Update obstacles from laser scan
    node._update_obstacles()

    # ── Waypoint progression ────────────────────────────────
    if node.current_waypoint_idx < len(node.waypoints):
        wx, wy = node.waypoints[node.current_waypoint_idx]
        dist_wp = math.hypot(node.x - wx, node.y - wy)
        if dist_wp < node.waypoint_tolerance:
            rospy.loginfo(
                'DWA: waypoint %d/%d reached '
                '(dist=%.2f m) — advancing to next.',
                node.current_waypoint_idx + 1, len(node.waypoints),
                dist_wp)
            node.current_waypoint_idx += 1

    # ── Active goal ─────────────────────────────────────────
    gx, gy = node._get_active_goal()
    orig_gx, orig_gy = node.goal_x, node.goal_y
    node.goal_x, node.goal_y = gx, gy
    v, omega = node._dwa_control()
    node.goal_x, node.goal_y = orig_gx, orig_gy

    # ── Apply velocity via SetModelState ────────────────────
    # (PlanarMovePlugin removed from URDF — crashes with
    #  boost::lock_error in headless mode)
    try:
        if _set_state_proxy is None:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
            from gazebo_msgs.srv import SetModelState
            _set_state_proxy = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
        dt = 0.1
        next_x = node.x + v * math.cos(node.yaw) * dt
        next_y = node.y + v * math.sin(node.yaw) * dt
        next_yaw = node.yaw + omega * dt
        q = Quaternion()
        q.z = math.sin(next_yaw / 2.0)
        q.w = math.cos(next_yaw / 2.0)
        state = ModelState()
        state.model_name = 'storm'
        state.reference_frame = 'world'
        state.pose.position.x = next_x
        state.pose.position.y = next_y
        state.pose.position.z = 0.05
        state.pose.orientation = q
        c, s = math.cos(next_yaw), math.sin(next_yaw)
        state.twist.linear.x = float(v) * c
        state.twist.linear.y = float(v) * s
        state.twist.angular.z = float(omega)
        _set_state_proxy(state)
    except Exception as e:
        rospy.logerr_throttle(5.0, 'DWA: SetModelState failed: %s', e)

    # Keep internal velocity estimate
    node.v = v
    node.omega = omega

    # ── Goal reached? ───────────────────────────────────────
    target_x = node.goal_x
    target_y = node.goal_y
    dist = math.hypot(node.x - target_x, node.y - target_y)
    if dist < 0.4:
        rospy.loginfo('DWA: goal reached (dist=%.2f m) — stopping.', dist)
        try:
            if _set_state_proxy is not None:
                stop = ModelState()
                stop.model_name = 'storm'
                stop.reference_frame = 'world'
                stop.twist.linear.x = 0.0
                stop.twist.linear.y = 0.0
                stop.twist.angular.z = 0.0
                q = Quaternion()
                q.z = math.sin(node.yaw / 2.0)
                q.w = math.cos(node.yaw / 2.0)
                stop.pose.position.x = node.x
                stop.pose.position.y = node.y
                stop.pose.position.z = 0.05
                stop.pose.orientation = q
                _set_state_proxy(stop)
        except Exception:
            pass
        break

    rate.sleep()

rospy.loginfo('DWA planner finished.')
