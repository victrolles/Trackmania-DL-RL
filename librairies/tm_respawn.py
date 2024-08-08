import numpy as np

from tminterface.interface import TMInterface

from librairies.data_classes import Point2D, SpawnConfig
from librairies.tm_math_functions import distance_to_finish_line, get_closest_point

class TMRandomRespawn:

    def __init__(self, spawn_config: SpawnConfig, middle_points: list[Point2D]):
        self.has_respawned = False
        self.has_get_start_dist = False
        self.middle_points = middle_points
        self.spawn_config = spawn_config

    def respawn(self, iface: TMInterface, _time: int):
        if self.has_respawned and _time > 100 and _time < 1000 :
            self.has_respawned = False
            random_int = np.random.randint(0, self.spawn_config.number)
            string = f"load_state checkpoint_{random_int}.bin"
            iface.execute_command(string)
            

        if not self.has_respawned and self.has_get_start_dist and _time > 300 :
            self.has_get_start_dist = False
            iface_state = iface.get_simulation_state()
            car_pos = Point2D(iface_state.position[0], iface_state.position[2])
            self.start_dist_to_finish_line = distance_to_finish_line(car_pos, get_closest_point(car_pos, self.middle_points), self.middle_points)

    def set_random_spawn(self):
        self.has_respawned = True
        self.has_get_start_dist = True