from dataclasses import dataclass

@dataclass
class Pose2D:
    x: float
    y: float
    angle_deg: float