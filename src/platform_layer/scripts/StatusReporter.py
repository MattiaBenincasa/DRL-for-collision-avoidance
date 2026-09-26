import rospy
import numpy as np
import math

class StatusReporter:
    def __init__(self, laser_sensor=None, state_size=50, collision_threshold=0.2):
        self.laser_sensor = laser_sensor
        self.lidar_size = int(state_size)
        self.collision_threshold = float(collision_threshold)
        self.state_size = self.lidar_size + 2

    def get_state(self, robot_pose=None, target_pose=None):
        data = getattr(self.laser_sensor, 'scan_data', None)

        if data is None or len(data) == 0:
            lidar = np.full(self.lidar_size, 1.0)
        else:
            idx = np.linspace(0, len(data) - 1, self.lidar_size).astype(int)
            lidar = np.asarray(data)[idx] / 5.0

        state_components = [lidar]

        if robot_pose is not None and target_pose is not None:
            dx = target_pose[0] - robot_pose.x
            dy = target_pose[1] - robot_pose.y
            dist = np.hypot(dx, dy) / 10.0
            angle = math.atan2(dy, dx) - math.radians(robot_pose.angle_deg)
            angle = math.atan2(math.sin(angle), math.cos(angle))
            state_components.append(np.array([dist, angle], dtype=np.float32))

        return np.concatenate(state_components).astype(np.float32)

    def is_collided(self):
        data = getattr(self.laser_sensor, 'scan_data', None)
        
        if data is None or len(data) == 0:
            return False

        return float(np.min(data)) < self.collision_threshold