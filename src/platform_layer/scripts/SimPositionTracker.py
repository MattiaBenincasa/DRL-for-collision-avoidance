import math

class SimPositionTracker:
    def __init__(self, simulation_manager):
        self.simulation_manager = simulation_manager
        self.start_position = None

    def get_position(self):
        return self.simulation_manager.get_position("storm")

    def set_start_position(self, start_position):
        self.start_position = start_position
