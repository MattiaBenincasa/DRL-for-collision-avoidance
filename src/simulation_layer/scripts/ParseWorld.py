import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import os
import math
import random

from Pose2D import Pose2D

@dataclass
class Zone:
    name: str
    zone_type: str
    x: float
    y: float
    yaw: float
    shape: str = 'rect'
    sx: Optional[float] = None
    sy: Optional[float] = None
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def size(self) -> Optional[Tuple[float, float]]:
        if self.sx is not None and self.sy is not None:
            return (self.sx, self.sy)
        return None

    def _local_to_world(self, lx: float, ly: float) -> Tuple[float, float]:
        cosy = math.cos(self.yaw)
        siny = math.sin(self.yaw)
        wx = self.x + lx * cosy - ly * siny
        wy = self.y + lx * siny + ly * cosy
        return (wx, wy)

    def sample_point(self, is_parallel: bool = False) -> Pose2D:
        if self.sx is not None and self.sy is not None:
            ux = random.uniform(-self.sx / 2.0, self.sx / 2.0)
            uy = random.uniform(-self.sy / 2.0, self.sy / 2.0)
        else:
            ux = 0.0
            uy = 0.0

        wx, wy = self._local_to_world(ux, uy)

        if is_parallel:
            long_size = max(self.sx, self.sy)
            is_long_x = self.sx >= self.sy

            if is_long_x:
                toward_center_angle = 0.0
                toward_edge_angle = 180.0
            else:
                toward_center_angle = 90.0
                toward_edge_angle = -90.0

            angle_deg = random.choice([toward_center_angle, toward_edge_angle])

            angle_deg = (angle_deg + math.degrees(self.yaw)) % 360.0
        else:
            angle_deg = random.uniform(0.0, 360.0)
        return Pose2D(wx, wy, angle_deg)

@dataclass
class MapInfo:
    name: str
    spawns: Dict[str, Zone]
    targets: Dict[str, Zone]
    spawn_to_target: Dict[str, Optional[str]]
    random_spawn_target: bool = False


class ParseWorld:
    def __init__(self, world_name: str, map_name: Optional[str] = None):
        # If world_name is already an existing file, use it directly;
        # otherwise treat it as a short name and resolve to worlds/.
        if os.path.exists(world_name):
            self.world_path = os.path.abspath(world_name)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.world_path = os.path.join(script_dir, "..", "worlds", f"{world_name}.world")
            self.world_path = os.path.abspath(self.world_path)
        self.tree = ET.parse(self.world_path)
        self.root = self.tree.getroot()
        self.map_name = map_name
        # Parse the <state> element to get the actual Gazebo positions
        self._state_poses: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self._parse_state()

    def _parse_pose(self, elem) -> Tuple[float, float, float]:
        p = elem.find("pose")
        if p is not None and p.text:
            toks = p.text.strip().split()
            x = float(toks[0]) if len(toks) > 0 else 0.0
            y = float(toks[1]) if len(toks) > 1 else 0.0
            yaw = float(toks[5]) if len(toks) > 5 else 0.0
            return (x, y, yaw)
        return (0.0, 0.0, 0.0)

    def _parse_state(self) -> None:
        """Parse the <state> element to get actual model/link positions as used by Gazebo.

        In Gazebo .world files, a <state> element stores the last saved pose of
        every model and link in absolute world coordinates.  When Gazebo loads
        the world it applies these poses on top of the SDF definitions, so the
        *state* coordinates are the real positions the spawn/target zones will
        actually have in the simulation.

        This method builds a lookup table:
            self._state_poses[(model_name, link_name)] -> (x, y, yaw)
        where (x, y, yaw) are absolute world coordinates.
        """
        world = self.root.find("world")
        if world is None:
            return
        state = world.find("state")
        if state is None:
            return

        for sm in state.findall("model"):
            mname = sm.attrib.get("name", "")
            for sl in sm.findall("link"):
                lname = sl.attrib.get("name", "")
                p = sl.find("pose")
                if p is not None and p.text:
                    toks = p.text.strip().split()
                    x = float(toks[0]) if len(toks) > 0 else 0.0
                    y = float(toks[1]) if len(toks) > 1 else 0.0
                    yaw = float(toks[5]) if len(toks) > 5 else 0.0
                    self._state_poses[(mname, lname)] = (x, y, yaw)

    def _lookup_state_pose(self, model_name: str, link_name: str) -> Optional[Tuple[float, float, float]]:
        """Look up the absolute world pose of a link from the <state>.

        Handles small naming inconsistencies between the SDF definition and
        the <state> (e.g. 'spawn_1a' vs 'spawn_1_a') by trying a normalized
        comparison (stripping underscores) when an exact match fails.
        """
        # 1. Exact match
        key = (model_name, link_name)
        if key in self._state_poses:
            return self._state_poses[key]

        # 2. Normalised match (strip underscores from both keys)
        norm_link = link_name.replace("_", "")
        for (mn, ln), pose in self._state_poses.items():
            if mn == model_name and ln.replace("_", "") == norm_link:
                return pose

        # 3. Also check if the model name has a different underscore pattern
        norm_model = model_name.replace("_", "")
        for (mn, ln), pose in self._state_poses.items():
            if mn.replace("_", "") == norm_model and ln.replace("_", "") == norm_link:
                return pose

        return None

    def find_zones(self, zone_type: str, model_name: Optional[str] = None) -> Dict[str, Zone]:
        found: Dict[str, Zone] = {}
        world = self.root.find("world")
        if world is None:
            return found

        if not model_name:
            for model in world.findall("model"):
                if zone_type in model.attrib.get("name", ""):
                    model_name = model.attrib["name"]
                    break
        if not model_name:
            return found

        model = world.find(f"model[@name='{model_name}']")
        if model is None:
            return found

        mx, my, myaw = self._parse_pose(model)

        for link in model.findall(".//link"):
            name = link.attrib.get("name", "unknown")
            lx, ly, lyaw = self._parse_pose(link)

            cos_myaw = math.cos(myaw)
            sin_myaw = math.sin(myaw)
            # Compute world position from SDF definition (model + local link pose)
            wx = mx + lx * cos_myaw - ly * sin_myaw
            wy = my + lx * sin_myaw + ly * cos_myaw
            wyaw = myaw + lyaw

            # Override with <state> absolute pose when available (Gazebo's real position)
            state_pose = self._lookup_state_pose(model_name, name)
            if state_pose is not None:
                wx, wy, wyaw_state = state_pose
                # Preserve yaw from state if non-zero, otherwise keep SDF yaw
                if wyaw_state != 0.0:
                    wyaw = wyaw_state

            sx = sy = None
            for vis in link.findall("visual"):
                size_elem = vis.find("geometry/box/size")
                if size_elem is not None and size_elem.text:
                    sizes = size_elem.text.strip().split()
                    sx = float(sizes[0])
                    sy = float(sizes[1]) if len(sizes) > 1 else sx
                    break
            if sx is None or sy is None:
                for col in link.findall("collision"):
                    size_elem = col.find("geometry/box/size")
                    if size_elem is not None and size_elem.text:
                        sizes = size_elem.text.strip().split()
                        sx = float(sizes[0])
                        sy = float(sizes[1]) if len(sizes) > 1 else sx
                        break
            if sx is not None and sy is not None:
                found[name] = Zone(name, zone_type, wx, wy, wyaw, 'rect', sx, sy)
        return found

    def get_map(self, map_name: str) -> MapInfo:
        spawn_model_name = f"{map_name}_spawn"
        target_model_name = f"{map_name}_target"

        spawns = self.find_zones('spawn', spawn_model_name)
        targets = self.find_zones('target', target_model_name)

        spawn_to_target: Dict[str, Optional[str]] = {}
        for sname in spawns.keys():
            m = re.search(r'(\d+)', sname)
            idx = m.group(1) if m else None
            matched = None
            if idx is not None:
                for tname in targets.keys():
                    if idx in re.split(r'_', tname):
                        matched = tname
                        break
            spawn_to_target[sname] = matched

        random_spawn = False
        world = self.root.find("world")
        walls = world.find(f"model[@name='{map_name}']") if world is not None else None
        if walls is not None:
            rs = walls.find('randomSpawnTarget')
            if rs is not None and rs.text:
                val = rs.text.strip().lower()
                random_spawn = val in ('1', 'true', 'yes', 'on')
        return MapInfo(
            name=map_name,
            spawns=spawns,
            targets=targets,
            spawn_to_target=spawn_to_target,
            random_spawn_target=random_spawn
        )

    def get_step_limits(self) -> Dict[str, int]:
        limits: Dict[str, int] = {}
        world = self.root.find("world")
        if world is None:
            return limits
        for model in world.findall("model"):
            name = model.attrib.get("name", "")
            sl = model.find("stepLimit")
            if sl is not None and sl.text:
                try:
                    limits[name] = int(sl.text.strip())
                except ValueError:
                    pass
        return limits
