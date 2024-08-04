import time

from tminterface.interface import TMInterface

from librairies.data_classes import TMSimulationResult, Point2D, Reward_limits
from librairies.tm_math_functions import distance_to_finish_line, distance_point_to_line, angle_between_lines, get_closest_point, get_next_point, get_road_points
from librairies.dictionaries import Rd

class SimulationRewards:

    def __init__(self, is_track_finished):

        self.middle_points = get_road_points(Rd.MIDDLE, True)
        self.reward_limits = Reward_limits(-1, 1)
        self.inactivity = 0
        self.is_track_finished = is_track_finished

# get reward, done, score        
    def get_TM_simulation_result(self, iface_state: TMInterface, car_pos: Point2D, car_ahead: Point2D, start_simulation_time: float) -> TMSimulationResult:
        
        reward = 0
        game_over = False
        
        # Get reward from speed
        speed = iface_state.display_speed
        reward += int(speed/20)
        if speed < 5:
            reward -= 3
        elif speed > 40:
            reward += 3
        elif speed > 60:
            reward += 5
        else:
            reward += 1
        
        closest_point = get_closest_point(car_pos, self.middle_points)

        previous_closest_point = get_next_point(closest_point, self.middle_points)

        dist_to_finish_line = distance_to_finish_line(car_pos, closest_point, self.middle_points)

        distance_to_center_line = distance_point_to_line(car_pos, previous_closest_point, closest_point)

        angle_to_center_line = angle_between_lines(previous_closest_point, closest_point, car_pos, car_ahead)

        if angle_to_center_line > 0.5:
            reward -= 3
        elif angle_to_center_line > 0.2:
            reward += 5
        else:
            reward += 3

        if distance_to_center_line < 3:
            reward += 5
        elif distance_to_center_line < 6:
            reward += 3
        elif distance_to_center_line < 9:
            reward += 1
        else:
            reward -= 3

        # if dist_to_finish_line < self.previous_dist_to_finish_line:
        #     reward += 7
        # else:
        #     reward -= 2
        self.dist_to_finish_line = dist_to_finish_line
        # reward += (TRACK_LENGTH - dist_to_finish_line) / 20

        # Get reward from getting closer to the middle line
        #TODO: Get the distance to the middle line

        # Get reward if no lateral contact
        if iface_state.scene_mobil.has_any_lateral_contact:
            game_over = True
            reward -= 10
        else:
            reward += 5

        # --- get DONE ---

        # Restart if too long
        if time.time() - start_simulation_time > 30:
            # print(time.time() - self.start_simulation_time)
            print("Step restarted : Car is too slow", flush=True)
            game_over = True
            reward -= 5

        # Restart if the car is stopped
        if speed < 5:
            self.inactivity += 1
        else:
            self.inactivity = 0
        # print(self.inactivity, flush=True)

        if self.inactivity > 10:
            print("Step restarted : Car is stopped", flush=True)
            game_over = True
            reward -= 5

        # Restart if the track is finished
        if self.is_track_finished:
            self.is_track_finished = False

            print("Step restarted : Track is finished")
            game_over = True
            reward += 10

        if game_over:
            self.inactivity = 0

        if reward < self.reward_limits.min:
            self.reward_limits.min = reward
            reward = -1
        elif reward > self.reward_limits.max:
            self.reward_limits.max = reward
            reward = 1
        elif reward < 0:
            reward = reward / self.reward_limits.min
        else:
            reward = reward / self.reward_limits.max

        return TMSimulationResult(reward, game_over, dist_to_finish_line)