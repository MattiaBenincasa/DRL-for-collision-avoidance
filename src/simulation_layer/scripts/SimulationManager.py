#!/usr/bin/env python
import os
import math
import random
import re
from typing import Optional, List, Dict
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState as _ModelState
from geometry_msgs.msg import Quaternion as _Quaternion
from std_srvs.srv import Empty
from ParseWorld import ParseWorld, Zone, MapInfo
from Pose2D import Pose2D

class SimulationManager:
    """Manages simulation spawning and goal checking for multiple maps in a single world.

    - Loads all maps named 'map_N' from the world file via ParseWorld.get_map
    - Does not keep an active map; all structured maps are kept in self.maps
    - respawn(map_name_or_index=None) will respawn the robot in a random spawn of the chosen map
    """

    def __init__(self, worldName: str):
        self.worldName = worldName
        self.last_model_states = None
        self._rospy = rospy
        # create proxies/subscribers; errors will bubble up
        self._set_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._resume_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._sub_model_states = rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_states_callback)
        self._ModelState = _ModelState
        self._Quaternion = _Quaternion

        # Parse world and load all map_* models
        self.pw = ParseWorld(worldName)
        self.maps: Dict[str, MapInfo] = {}
        world = self.pw.root.find('world')
        for m in world.findall('model') if world is not None else []:
            name = m.attrib.get('name', '')
            if re.match(r'^map_\d+$', name):
                try:
                    self.maps[name] = self.pw.get_map(name)
                except Exception as e:
                    if self._rospy:
                        rospy.logwarn(f"Failed to parse map '{name}': {e}")
                    else:
                        print(f"Warning: Failed to parse map '{name}': {e}")

        world_basename = os.path.splitext(os.path.basename(self.pw.world_path))[0]
        if world_basename.startswith('map_') and world_basename not in self.maps:
            try:
                self.maps[world_basename] = self.pw.get_map(world_basename)
            except Exception as e:
                if self._rospy:
                    rospy.logwarn(f"Failed to parse map '{world_basename}': {e}")
                else:
                    print(f"Warning: Failed to parse map '{world_basename}': {e}")

        if not self.maps:
            raise ValueError(f"No structured maps (map_N) found in world {worldName}")

        # per-map time limits
        try:
            self.map_step_limits = self.pw.get_step_limits()
        except Exception as e:
            self.map_step_limits = {}
            print(f"Warning: failed to read map step limits: {e}")
        # current_map and step_limit will be set when respawn selects a map
        self.current_map = None
        self.step_limit = None
        # Last selected spawn (updated by respawn)
        self.last_spawn = None

    def _model_states_callback(self, msg):
        self.last_model_states = msg

    def list_maps(self) -> List[str]:
        return sorted(self.maps.keys())

    def get_spawn(self, map_name_or_index) -> List[Zone]:
        """Return (list_of_spawn_zones) for the given map.

        map_name_or_index can be either a string 'map_1' or an integer 1.
        If the map has multiple spawn zones that share the same numeric index,
        the returned spawn list will contain all of them (e.g., ['spawn_1a','spawn_1b']).

        The pairing policy (random vs matching indices) is determined by the
        map's metadata tag <randomSpawnTarget> parsed by ParseWorld.

        Raises ValueError if the map is not available or structured spawn/target data is missing.
        """
        # normalize map name
        if isinstance(map_name_or_index, int):
            map_name = f"map_{map_name_or_index}"
        elif isinstance(map_name_or_index, str):
            if map_name_or_index in self.maps:
                map_name = map_name_or_index
            elif re.match(r'^map_\d+$', map_name_or_index):
                map_name = map_name_or_index
            else:
                m = re.search(r"(\d+)", map_name_or_index)
                if m:
                    candidate = f"map_{int(m.group(1))}"
                    if candidate in self.maps:
                        map_name = candidate
                    else:
                        raise ValueError(f"Map '{map_name_or_index}' not available")
                else:
                    raise ValueError(f"Invalid map identifier: {map_name_or_index}")
        else:
            raise ValueError("map_name_or_index must be str or int")

        if map_name not in self.maps:
            raise ValueError(f"Map '{map_name}' not available")

        mapinfo = self.maps[map_name]

        if not mapinfo.spawns:
            raise ValueError(f"Map '{map_name}' does not have any spawn zones")

        spawn_objs = list(mapinfo.spawns.values())
        return spawn_objs

    def respawn(self, parallel_spawn: bool = False, spawn_index: Optional[int] = None):
        """Respawn the robot into a spawn zone.

        - Choose a random map from available maps.
        - Use get_spawn to obtain spawn zone(s).
        - If multiple spawn zones are returned, pick one at random, then sample a random point
          inside that spawn (orientation is randomized).
        - If ROS is available, move the robot in Gazebo; otherwise return the chosen pose.
        """
        # choose a random map
        map_name = random.choice(list(self.maps.keys()))

        # set current map and associated time limit (default 500s if missing)
        self.current_map = map_name
        self.step_limit = self.map_step_limits.get(map_name, 1000)

        # obtain spawn(s) for the chosen map
        spawns = self.get_spawn(map_name)
        if not isinstance(spawns, (list, tuple)):
            spawns = [spawns]

        if spawn_index is not None:
            matching = [z for z in spawns if re.search(rf'_{spawn_index}(?:[a-z]|$)', getattr(z, 'name', ''))]
            if matching:
                zone = random.choice(matching)
            else:
                zone = random.choice(spawns) if spawns else None
        else:
            zone = random.choice(spawns) if spawns else None

        if zone is None:
            raise ValueError(f"No spawn zones available for map '{map_name}'")

        # debug info: show selected zone and its parameters
        try:
            zone_name = getattr(zone, 'name', str(zone))
            cx, cy = zone.center if zone.center else (None, None)
            zsize = zone.size
            zyaw = getattr(zone, 'yaw', None)
            if self._rospy:
                rospy.loginfo(f"[DEBUG] map={map_name} selected zone={zone_name} center=({cx},{cy}) size={zsize} yaw={zyaw}")
            else:
                print(f"[DEBUG] map={map_name} selected zone={zone_name} center=({cx},{cy}) size={zsize} yaw={zyaw}")
        except Exception:
            pass

        # sample a random point inside chosen spawn
        p = zone.sample_point(is_parallel=parallel_spawn)
        x = p.x
        y = p.y
        angle_deg = p.angle_deg

        # verify point is inside rectangle (if rect)
        try:
            inside = True
            if getattr(zone, 'shape', '') == 'rect' and zone.size is not None:
                cx, cy = zone.center
                dx = x - cx
                dy = y - cy
                cosy = math.cos(-zone.yaw)
                siny = math.sin(-zone.yaw)
                lx = dx * cosy - dy * siny
                ly = dx * siny + dy * cosy
                half_x = zone.size[0] / 2.0
                half_y = zone.size[1] / 2.0
                inside = (-half_x <= lx <= half_x) and (-half_y <= ly <= half_y)
            if self._rospy:
                rospy.loginfo(f"[DEBUG] sampled point: x={x}, y={y}, inside_rect={inside}")
            else:
                print(f"[DEBUG] sampled point: x={x}, y={y}, inside_rect={inside}")
        except Exception:
            pass

        # apply via ROS if available
        if self._rospy and self._set_state_proxy and self._ModelState and self._Quaternion:
            try:
                rospy = self._rospy
                q = self._Quaternion()
                angle_rad = math.radians(angle_deg)
                q.z = math.sin(angle_rad / 2.0)
                q.w = math.cos(angle_rad / 2.0)
                state_msg = self._ModelState()
                state_msg.model_name = 'storm'
                state_msg.pose.position.x = float(x)
                state_msg.pose.position.y = float(y)
                state_msg.pose.position.z = 0.05
                state_msg.pose.orientation = q
                rospy.wait_for_service('/gazebo/set_model_state')
                self._set_state_proxy(state_msg)
                #rospy.loginfo("Robot respawned at: x={}, y={}, angle={}".format(round(x,3), round(y,3), round(angle_deg,2)))
            except Exception as e:
                if self._rospy:
                    rospy.logerr(f"Error SetModelState: {e}")
                else:
                    print(f"Error SetModelState: {e}")
        else:
            # No ROS available; return the chosen pose for test/inspection
            if not self._rospy:
                print(f"[SimulationManager] Selected spawn in {map_name}: x={x}, y={y}, angle={angle_deg}")

        # record last spawn for external inspection
        try:
            self.last_spawn = zone
        except Exception:
            self.last_spawn = None

    def pause_simulation(self) -> None:
        try:
            rospy.wait_for_service('/gazebo/pause_physics', timeout=3.0)
            self._pause_proxy()
        except Exception as e:
            rospy.logerr(f"Failed to pause simulation: {e}")

    def resume_simulation(self) -> None:
        try:
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=3.0)
            self._resume_proxy()
        except Exception as e:
            rospy.logerr(f"Failed to resume simulation: {e}")

    def get_position(self, entity_name: str) -> Optional[Pose2D]:
        """Return absolute position of entity_name as Pose2D (x, y, angle_deg)."""
        if self.last_model_states is None:
            if self._rospy:
                self._rospy.logwarn("Model states not received yet.")
            else:
                print("Model states not received yet.")
            return None

        if entity_name in self.last_model_states.name:
            idx = self.last_model_states.name.index(entity_name)
            pose = self.last_model_states.pose[idx]
            x = pose.position.x
            y = pose.position.y
            try:
                q = pose.orientation
                # compute yaw (rotation around Z) from quaternion
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                angle_deg = math.degrees(yaw)
            except Exception:
                angle_deg = 0.0
            return Pose2D(x, y, angle_deg)
        else:
            if self._rospy:
                self._rospy.logwarn("Entity %s not found in /gazebo/model_states", entity_name)
            else:
                print(f"Entity {entity_name} not found in model_states")
            return None