"""
DWA (Dynamic Window Approach) local planner — ROS node for the DLRL LIDAR robot.

Adapted from the standalone Pygame version to work within the project's ROS
architecture.

  - Subscribes to /scan (LaserScan) to build obstacle map
  - Subscribes to /gazebo/model_states (ModelStates) for robot pose
  - Applies (v, ω) via /gazebo/set_model_state (kinematic control —
    bypasses PlanarMovePlugin which crashes in headless mode)
  - Goal from ParseWorld (map_1.world target zone) — no override
  - Waypoints from private param set by DWA.launch (fixed list)

DWA algorithm:
  1. Build the dynamic window from current velocity and acceleration limits.
  2. Sample velocity pairs (v, ω) within the window.
  3. Forward-simulate a trajectory for each pair (predict_time seconds).
  4. Score each trajectory with: obstacle_cost + to_goal_cost + speed_cost.
  5. Execute the velocity command with the lowest cost.
"""

import os
import math
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, SpawnModel

# Project-layer imports (following the same pattern as testing.py / training.py)
from platform_layer.scripts.LaserSensor import LaserSensor
from platform_layer.scripts.StatusReporter import StatusReporter
from simulation_layer.scripts.ParseWorld import ParseWorld


# ══════════════════════════════════════════════════════════════
#  DWA configuration
# ══════════════════════════════════════════════════════════════

class Config:
    """Tuning parameters for the DWA planner — adapted for the storm robot."""

    def __init__(self):
        # ── Velocity limits ──
        self.max_speed = 0.5                     # m/s
        self.min_speed = 0.0                     # m/s  (no reverse)
        self.max_omega = 0.8                     # rad/s (matches CommandManager
                                                   #   range: -0.8…+0.8)

        # ── Acceleration limits ──
        self.max_accel = 1.0                     # m/s²  (higher = faster
                                                   #    acceleration from standstill)
        self.max_omega_accel = 3.0               # rad/s²  (fast turning response)

        # ── Sampling resolution ──
        self.v_resolution = 0.05                 # m/s step
        self.omega_resolution = 1.0 * math.pi / 180.0  # rad/s step ≈ 1°

        # ── Prediction horizon ──
        self.dt = 0.2                            # integration step (s)
        self.predict_time = 4.0                  # total horizon (s); waypoints are
                                                   # 1-2 m apart, so at 0.4 m/s the
                                                   # robot looks ≈1.6 m ahead —
                                                   # enough to reach next waypoint
                                                   # without overshooting badly

        # ── Robot geometry ──
        self.robot_radius = 0.3                  # m (slightly larger than
                                                  #    the 0.4×0.3 m base)

        # ── Cost weights ──
        self.to_goal_cost_gain  = 5.0    # moderate pull toward goal
        self.speed_cost_gain    = 5.0    # strongly encourage moving fast
        self.obstacle_cost_gain = 0.5    # reduced for narrow corridor navigation
                                           #   (map_1 corridor is only 1.24 m wide)

        # ── Laser obstacle extraction ──
        self.laser_max_range = 4.0               # ignore points beyond (m)
        self.laser_min_range = 0.5               # ignore points closer (m); 0.5 =
                                                   #   robot_radius(0.3) + margin
                                                   #   filters out self-detection
        self.obstacle_radius = 0.15              # inflated radius per point


# ══════════════════════════════════════════════════════════════
#  ROS node
# ══════════════════════════════════════════════════════════════

class DWA:
    """ROS node that runs the Dynamic Window Approach local planner."""

    def __init__(self):
        self.cfg = Config()

        # ── Robot state (updated from /gazebo/model_states) ──
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.pose_ready = False

        # ── Goal position (from ParseWorld, no override) ──
        self.goal_x = None
        self.goal_y = None

        # ── Waypoint navigation (fixed list from .launch) ──
        waypoints_str = rospy.get_param('~waypoints', '')
        self.waypoints = []
        if waypoints_str:
            for pair in waypoints_str.split(';'):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    x_str, y_str = pair.split(',')
                    self.waypoints.append((float(x_str), float(y_str)))
                except (ValueError, IndexError):
                    rospy.logwarn('DWA: invalid waypoint pair "%s"', pair)
            if self.waypoints:
                rospy.loginfo('DWA: loaded %d waypoints', len(self.waypoints))
                rospy.loginfo('DWA: waypoint route: %s',
                              ' → '.join(f'({wp[0]:.1f},{wp[1]:.1f})' for wp in self.waypoints))
            else:
                rospy.loginfo('DWA: no valid waypoints — going directly to goal')
        else:
            rospy.loginfo('DWA: no waypoints specified — going directly to goal')

        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.4  # m — advance when within this distance
        self.waypoint_reached = False

        # ── Obstacles (rebuilt from every laser scan) ──
        self.obstacles = []          # list of (x, y, radius) in world frame

        # ── Sensor stack (laser via StatusReporter, per project architecture) ──
        self.laser_sensor = LaserSensor()
        self.status_reporter = StatusReporter(laser_sensor=self.laser_sensor)
        # cmd_vel no longer used: PlanarMovePlugin removed from URDF.
        # Kinematic control via /gazebo/set_model_state.

        # ── Additional subscriptions ──
        self.sub_states = rospy.Subscriber('/gazebo/model_states',
                                           ModelStates, self._cb_states)
        self.sub_goal = rospy.Subscriber('/goal', Point, self._cb_goal)

        # ── ParseWorld setup (spawn & target from world file) ──
        self._setup_from_parseworld()

        # Service proxy to bypass PlanarMovePlugin (crashes with
        # boost::lock_error in headless mode). Created lazily on
        # the first control cycle because the service may not be
        # ready yet.
        self._set_state_proxy = None

    # ── Callbacks ───────────────────────────────────────────

    def _cb_goal(self, msg: Point):
        self.goal_x = msg.x
        self.goal_y = msg.y
        rospy.loginfo('DWA: goal updated to (%.2f, %.2f)', self.goal_x, self.goal_y)

    def _cb_states(self, msg: ModelStates):
        """Extract pose AND actual velocity of the 'storm' robot from Gazebo."""
        if 'storm' not in msg.name:
            return
        idx = msg.name.index('storm')
        pose = msg.pose[idx]
        self.x = pose.position.x
        self.y = pose.position.y
        # quaternion → yaw (Euler Z)
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        # Actual velocity from Gazebo (used for dynamic window computation
        # so DWA's internal estimate matches real robot state even if
        # CommandManager applies a different velocity).
        twist = msg.twist[idx]
        self.v = twist.linear.x
        self.omega = twist.angular.z
        self.pose_ready = True

    # ── DWA core algorithm ─────────────────────────────────

    @staticmethod
    def _predict_trajectory(x0, y0, yaw0, v, omega, cfg):
        """Forward-simulate a trajectory for one velocity pair (pre-allocated)."""
        n_steps = int(cfg.predict_time / cfg.dt) + 1
        traj = np.zeros((n_steps, 5))
        traj[0] = [x0, y0, yaw0, v, omega]
        x, y, yaw = x0, y0, yaw0
        for i in range(1, n_steps):
            yaw += omega * cfg.dt
            x   += v * math.cos(yaw) * cfg.dt
            y   += v * math.sin(yaw) * cfg.dt
            traj[i] = [x, y, yaw, v, omega]
        return traj

    def _obstacle_cost(self, trajectory):
        """Return 1 / min_distance  (inf if any future point collides).

        The first trajectory point (current robot pose) is SKIPPED because
        the robot is already there — obstacle points that happen to be within
        collision distance of the starting pose are self-detection noise or
        ground returns that should not block all future motion.

        Uses numpy broadcasting for vectorised distance computation.
        """
        start_i = 1 if len(trajectory) > 1 else 0  # skip current pose
        n_pts = len(trajectory) - start_i
        if n_pts == 0 or not self.obstacles:
            return 0.0

        threshold = self.cfg.robot_radius + self.cfg.obstacle_radius

        # Build numpy arrays for broadcasting
        tx = trajectory[start_i:, 0]
        ty = trajectory[start_i:, 1]
        obs_arr = np.array(self.obstacles)  # (M, 3) — each row: ox, oy, r

        # Broadcast: (n_pts, 1) - (1, M) → (n_pts, M)
        dx = tx[:, None] - obs_arr[None, :, 0]
        dy = ty[:, None] - obs_arr[None, :, 1]
        dists = np.hypot(dx, dy)  # (n_pts, M)

        # Collision check
        if np.any(dists <= threshold):
            return float('inf')

        min_d = float(dists.min())
        if min_d < threshold * 1.5:
            rospy.loginfo_throttle(2.0,
                'DWA: min obstacle dist=%.3f (thr=%.3f)',
                min_d, threshold)
        return 1.0 / max(min_d, 0.01)

    @staticmethod
    def _goal_cost(trajectory, gx, gy):
        """Angular error + progress-to-goal cost.

        Returns a combined measure:
          - angular error  between final heading and direction to goal
          - progress cost: 0 = waypoint reached, 1 = no progress made,
            capped at 1.0 so overshooting is NOT penalised worse than
            standing still (otherwise the DWA prefers v→0 near close
            waypoints, causing the famous "molto lento" problem).
        """
        tx, ty = trajectory[-1, 0], trajectory[-1, 1]
        dx = gx - tx
        dy = gy - ty
        dist = math.hypot(dx, dy)

        # Angular cost (same as before)
        goal_angle = math.atan2(dy, dx)
        err = goal_angle - trajectory[-1, 2]
        angle_cost = abs(math.atan2(math.sin(err), math.cos(err)))

        # Progress cost: fraction of initial distance remaining,
        # CAPPED at 1.0 so overshoot ≠ penalty.
        x0, y0 = trajectory[0, 0], trajectory[0, 1]
        initial_dist = math.hypot(gx - x0, gy - y0)
        if initial_dist > 0.01:
            dist_cost = min(dist / initial_dist, 1.0)
        else:
            dist_cost = 0.0

        # Combine: angular error dominates near the goal, distance dominates far away.
        # Angle weight kept low so the robot can temporarily face away from the
        # goal when navigating around obstacles.
        return 0.05 * angle_cost + 0.95 * dist_cost

    def _dwa_control(self):
        """Sample (v, ω) in the dynamic window; return best (v, ω).

        Velocity is applied externally via SetModelState (kinematic control
        in run_DWA.py).

        If no obstacles are visible (open space), steer gently toward the
        goal for efficiency.
        """
        # Dynamic window for v and ω (based on actual robot velocity from
        # Gazebo model_states callback)
        v_min = max(self.cfg.min_speed,
                    self.v - self.cfg.max_accel * self.cfg.dt)
        v_max = min(self.cfg.max_speed,
                    self.v + self.cfg.max_accel * self.cfg.dt)
        o_min = max(-self.cfg.max_omega,
                    self.omega - self.cfg.max_omega_accel * self.cfg.dt)
        o_max = min(self.cfg.max_omega,
                    self.omega + self.cfg.max_omega_accel * self.cfg.dt)

        # ── Fast path: open space ─────────────────────────
        if not self.obstacles:
            dx = self.goal_x - self.x
            dy = self.goal_y - self.y
            goal_angle = math.atan2(dy, dx)
            err = goal_angle - self.yaw
            err = math.atan2(math.sin(err), math.cos(err))
            omega_cmd = max(-self.cfg.max_omega,
                            min(self.cfg.max_omega, err * 0.5))
            return [v_max, omega_cmd]

        # ── Full search over v × ω ────────────────────────
        best_cost = float('inf')
        best_u = [0.0, 0.0]
        valid_count = 0
        total_count  = 0

        v_samples = np.arange(
            max(v_min, self.cfg.v_resolution), v_max, self.cfg.v_resolution)
        if len(v_samples) == 0:
            v_samples = np.array([self.cfg.v_resolution])

        for v in v_samples:
            for omega in np.arange(o_min, o_max, self.cfg.omega_resolution):
                total_count += 1
                traj = self._predict_trajectory(
                    self.x, self.y, self.yaw, v, omega, self.cfg
                )

                obs_cost  = self.cfg.obstacle_cost_gain * self._obstacle_cost(traj)
                gl_cost   = self.cfg.to_goal_cost_gain * self._goal_cost(traj,
                                self.goal_x, self.goal_y)
                spd_cost  = self.cfg.speed_cost_gain * (
                    self.cfg.max_speed - abs(traj[-1, 3]))

                cost = obs_cost + gl_cost + spd_cost
                if cost < float('inf'):
                    valid_count += 1
                if cost < best_cost:
                    best_cost = cost
                    best_u = [v, omega]

        rospy.loginfo_throttle(
            1.0,
            'DWA: v=[%.2f, %.2f] ω=[%.2f, %.2f], '
            'valid=%d/%d, best_cost=%s',
            v_min, v_max, o_min, o_max,
            valid_count, total_count,
            'inf' if best_cost==float('inf') else f'{best_cost:.2f}')

        # ── Recovery behavior ─────────────────────────────
        if best_cost == float('inf'):
            rospy.logwarn_throttle(3.0,
                'DWA: all trajectories blocked — rotating in place')
            dx = self.goal_x - self.x
            dy = self.goal_y - self.y
            goal_angle = math.atan2(dy, dx)
            err = goal_angle - self.yaw
            err = math.atan2(math.sin(err), math.cos(err))
            omega_rec = max(-self.cfg.max_omega,
                            min(self.cfg.max_omega, err * 1.5))
            return [v_min, omega_rec]

        return best_u

    # ── Action mapping ─────────────────────────────────────

    @staticmethod
    def _omega_to_action(omega):
        """Map continuous angular velocity (rad/s) to discrete action [0..10].

        CommandManager maps:  omega = -0.8 + 0.16 * action
        Inverse:              action = (omega + 0.8) / 0.16
        """
        action = int(round((omega + 0.8) / 0.16))
        return max(0, min(10, action))

    # ── Main loop ──────────────────────────────────────────

    def _update_obstacles(self):
        """Rebuild world-frame obstacle list from subsampled laser ranges.

        Subsampling every 10th ray (512 → ≈51) reduces obstacle count without
        missing critical walls, speeding up _obstacle_cost significantly.
        """
        raw = self.status_reporter.laser_sensor.scan_data
        if raw is None or len(raw) == 0:
            return

        n = len(raw)
        angle_min = -2.35619449   # storm URDF: −3π/4
        angle_max =  2.35619449   # storm URDF: +3π/4

        # Laser origin in world frame
        laser_off_x = 0.15
        laser_off_y = 0.0
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        lx_w = self.x + laser_off_x * c - laser_off_y * s
        ly_w = self.y + laser_off_x * s + laser_off_y * c

        # Subsample every 10th ray for speed (512 → ≈51)
        step = max(1, n // 50)
        sub_raw = np.asarray(raw[::step])
        sub_angles = np.linspace(angle_min, angle_max, len(sub_raw))

        # Range mask
        mask = (sub_raw > self.cfg.laser_min_range) & (sub_raw < self.cfg.laser_max_range)
        r_valid = sub_raw[mask]
        a_valid = sub_angles[mask]

        # Points in laser frame (vectorised)
        px = r_valid * np.cos(a_valid)
        py = r_valid * np.sin(a_valid)

        # Rotate to world frame
        wx = lx_w + px * c - py * s
        wy = ly_w + px * s + py * c
        self.obstacles = list(zip(wx, wy,
                                  [self.cfg.obstacle_radius] * len(wx)))

        # ── Debug: log obstacle count and closest distance ──
        if self.obstacles:
            obs_arr = np.array([[ox, oy] for ox, oy, _ in self.obstacles])
            dists = np.hypot(obs_arr[:, 0] - self.x,
                             obs_arr[:, 1] - self.y)
            closest_idx = int(dists.argmin())
            cx, cy = self.obstacles[closest_idx][:2]
            rospy.loginfo_throttle(
                1.0,
                'DWA: %d obstacles, closest=%.2f m at (%.2f, %.2f) '
                '(robot=(%.2f, %.2f))',
                len(self.obstacles), float(dists.min()), cx, cy,
                self.x, self.y)

    def _get_active_goal(self):
        """Return the current (goal_x, goal_y) considering waypoint progression.

        While waypoints remain, the active goal is the current waypoint.
        After all waypoints are reached, the original goal_x, goal_y is used.
        """
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return (self.goal_x, self.goal_y)

    # ── Spawn robot in Gazebo ──────────────────────────

    def _spawn_robot(self):
        """Spawn storm in Gazebo reading model from /robot_description.

        Waits for the /gazebo/spawn_urdf_model service to become available
        and for the simulation to be ready, avoiding race conditions
        at startup.
        """
        robot_desc = rospy.get_param('/robot_description', None)
        if robot_desc is None:
            rospy.logfatal('DWA: /robot_description not found!')
            return

        # Wait for spawn service
        rospy.loginfo('DWA: waiting for /gazebo/spawn_urdf_model ...')
        rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=30.0)

        try:
            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.05
            pose.orientation.w = 1.0

            spawn = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            resp = spawn('storm', robot_desc, '', pose, 'world')
            if resp.success:
                rospy.loginfo('DWA: storm spawned successfully')
            else:
                rospy.logwarn('DWA: spawn failed: %s', resp.status_message)

        except Exception as e:
            rospy.logerr('DWA: exception during spawn: %s', e)

    # ── ParseWorld integration (single source for spawn & goal) ──

    def _setup_from_parseworld(self):
        """Read the world file via ParseWorld for spawn and target/goal."""
        world_file = rospy.get_param('~world_file', '')
        if not world_file or not os.path.exists(world_file):
            rospy.logwarn('DWA: no world file — goal not set')
            self._pw_spawn_x = None
            return

        try:
            pw = ParseWorld(world_file)

            # ── Spawn zone ──
            spawn_zones = pw.find_zones('spawn')
            if spawn_zones:
                first = list(spawn_zones.values())[0]
                self._pw_spawn_x = first.x
                self._pw_spawn_y = first.y
                self._pw_spawn_yaw = first.yaw
                rospy.loginfo('DWA: ParseWorld spawn = (%.3f, %.3f, yaw=%.3f)',
                              first.x, first.y, first.yaw)
            else:
                rospy.logwarn('DWA: no spawn zone in %s', world_file)
                self._pw_spawn_x = None

            # ── Target / goal zone ──
            target_zones = pw.find_zones('target')
            if target_zones:
                first = list(target_zones.values())[0]
                self.goal_x = first.x
                self.goal_y = first.y
                rospy.loginfo('DWA: ParseWorld goal = (%.3f, %.3f)',
                              first.x, first.y)
            else:
                rospy.logwarn('DWA: no target zone in %s — using fallback',
                              world_file)
                self.goal_x = 0.44
                self.goal_y = -0.11

        except Exception as e:
            rospy.logwarn('DWA: ParseWorld failed: %s — using fallback', e)
            self._pw_spawn_x = None
            self.goal_x = 0.44
            self.goal_y = -0.11

    def _respawn_at_spawn(self):
        """Move robot to the ParseWorld spawn, oriented toward first waypoint."""
        pw_x = getattr(self, '_pw_spawn_x', None)
        if pw_x is None:
            rospy.loginfo('DWA: no ParseWorld spawn — keeping initial position')
            return

        try:
            # Point toward first waypoint, otherwise use spawn zone yaw
            if self.waypoints:
                wx, wy = self.waypoints[0]
                yaw = math.atan2(wy - self._pw_spawn_y, wx - self._pw_spawn_x)
            else:
                yaw = getattr(self, '_pw_spawn_yaw', 0.0)

            rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            q = Quaternion()
            q.z = math.sin(yaw / 2.0)
            q.w = math.cos(yaw / 2.0)
            state = ModelState()
            state.model_name = 'storm'
            state.reference_frame = 'world'
            state.pose.position.x = self._pw_spawn_x
            state.pose.position.y = self._pw_spawn_y
            state.pose.position.z = 0.05
            state.pose.orientation = q
            set_state(state)
            rospy.loginfo('DWA: respawned at (%.3f, %.3f, yaw=%.3f) toward waypoint',
                          self._pw_spawn_x, self._pw_spawn_y, yaw)
        except Exception as e:
            rospy.logerr('DWA: respawn failed: %s', e)



