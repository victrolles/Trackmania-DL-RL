import sys
import math
import json
import time

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import pandas as pd
import torch
import numpy as np

from config.data_classes import Point2D, DataBus, TMSimulationResult, Exp, RadarState, Point2D
from config.globals import TRACK_NAME, TRACK_LENGTH, BUFFER_SIZE, FRAME_COOLDOWN_ACTION, FRAME_COOLDOWN_DISPLAY_STATE, FRAME_COOLDOWN_DISPLAY_STATS, INITIAL_GAME_SPEED
from config.dictionaries import INPUT

from tm_agents.radar_agent import RadarAgent

from rl_algorithms.dqn.dqn_trainer import DQNTrainer
from rl_algorithms.experience_buffer import ExperienceBuffer

class Environment(Client):
    def __init__(self,
                 databus_buffer: DataBus,
                 end_processes,
                 tm_speed,
                 is_training,
                 saving_model,
                 is_map_render,
                 is_curves_render,
                 is_tm_speed_changed) -> None:
        super(Environment, self).__init__()

        ## Bus data
        self.databus_buffer = databus_buffer

        ## Shared memory
        self.end_processes = end_processes
        self.tm_speed = tm_speed
        self.is_training = is_training
        self.saving_model = saving_model
        self.is_map_render = is_map_render
        self.is_curves_render = is_curves_render
        self.is_tm_speed_changed = is_tm_speed_changed

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iter = 0
        self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        self.previous_dist_to_finish_line = TRACK_LENGTH
        self.previous_state = None
        self.previous_action = None
        self.inactivity = 0
        self.is_track_finished = False
        self.current_game_speed = INITIAL_GAME_SPEED
        self.training_stats = None
        self.start_total_time = time.time()

        
        list_points_left_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_left.csv')
        list_points_right_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_right.csv')
        list_points_left_border = list_points_left_border.iloc[::20]
        list_points_right_border = list_points_right_border.iloc[::20]
        self.list_points_left_border = list(zip(list_points_left_border.x_values, list_points_left_border.y_values))
        self.list_points_right_border = list(zip(list_points_right_border.x_values, list_points_right_border.y_values))
        print("size left: ", len(self.list_points_left_border))
        print("size right: ", len(self.list_points_right_border))
        self.list_points_middle_line = get_list_point_middle_line()
        self.road_sections = get_road_sections()

        
        self.agent = RadarAgent(self.list_points_left_border, self.list_points_right_border)
        self.dqn_trainer = DQNTrainer(self.experience_buffer, self.device)
        

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)

        iface.set_speed(self.current_game_speed)
        self.tm_speed.value = self.current_game_speed

        iface.set_timeout(40_000)
        iface.give_up()

    # Function to detect if the car crossed the finish line
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()
            self.is_track_finished = True
            iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time >= 0:
            self.iter += 1

            if self.iter % FRAME_COOLDOWN_ACTION == 0:

                # ===== Stop the process if needed =====
                if self.end_processes.value:
                    self.stop_env_process(iface)

                # ===== Change Training Speed =====
                if self.is_tm_speed_changed.value:
                    self.change_tm_speed(iface)

                # ===== Save the model if needed =====
                if self.saving_model.value:
                    self.save_model()

                iface_state = iface.get_simulation_state()
                
                # ===== Get the state =====
                state = self.agent.get_state(iface_state)
                # print(state.distances, flush=True)

                # ===== Get the TM simulation result =====
                tm_simulation_result = self.get_TM_simulation_result(iface_state, _time, state.car_pos, state.car_ahead_pos)

                # print(f"Reward: {tm_simulation_result.reward}", flush=True)

                # ===== Store experience =====
                if self.previous_state is not None and self.previous_action is not None:
                    if check_valid_RadarState(state):
                        experience = Exp(self.previous_state.distances,
                                        self.previous_action,
                                        tm_simulation_result.reward,
                                        state.distances,
                                        tm_simulation_result.done)
                        self.experience_buffer._append(experience)

                # ===== Get the current state of the car =====
                if tm_simulation_result.done:
                    self.previous_state = None
                    self.previous_action = None
                else:
                    self.previous_state = state
                    if iface_state.position[2] < 122:
                        self.previous_action = 0
                    else:
                        self.previous_action = self.agent.get_action(self.dqn_trainer.model_network,
                                                                    self.previous_state.distances,
                                                                    self.dqn_trainer.epsilon,
                                                                    self.device,
                                                                    self.is_training.value)
                    # print(self.previous_action, flush=True)
                    iface.set_input_state(**INPUT[self.previous_action])

                if self.iter % FRAME_COOLDOWN_DISPLAY_STATE == 0 and self.is_map_render.value:
                    self.databus_buffer.put(DataBus(state,
                                                    None,
                                                    time.time() - self.start_total_time,
                                                    0,
                                                    self.iter/(time.time() - self.start_total_time),
                                                    len(self.experience_buffer)))

                # ===== Update the model if game over =====
                if tm_simulation_result.done and len(self.experience_buffer) > 0:
                    iface.give_up()

                    self.training_stats = self.dqn_trainer.train_model()

                    if self.iter % FRAME_COOLDOWN_DISPLAY_STATS == 0 and self.is_curves_render.value:
                        self.databus_buffer.put(DataBus(None,
                                                        self.training_stats,
                                                        time.time() - self.start_total_time,
                                                        TRACK_LENGTH - self.previous_dist_to_finish_line,
                                                        self.iter/(time.time() - self.start_total_time),
                                                        len(self.experience_buffer)))
                    
                    iface.give_up()

                    self.previous_dist_to_finish_line = TRACK_LENGTH
                    self.inactivity = 0

    def stop_env_process(self, iface: TMInterface) -> None:
        # Close the connection to Trackmania
        iface.close()
        # Save the model
        self.dqn_trainer.save_model()
        print("Environment process correctly stopped", flush=True)
        exit()

    def change_tm_speed(self, iface: TMInterface) -> None:
        self.current_game_speed = self.tm_speed.value
        iface.set_speed(self.tm_speed.value)
        print(f"Speed changed to {self.tm_speed.value}", flush=True)
        self.is_tm_speed_changed.value = False

    def save_model(self) -> None:
        self.dqn_trainer.save_model()
        print("Models correctly saved", flush=True)
        self.saving_model.value = False

    # get reward, done, score        
    def get_TM_simulation_result(self, iface_state: TMInterface, _time: int, car_pos: Point2D, car_ahead: Point2D) -> TMSimulationResult:
        
        reward = 0
        game_over = False
        
        # Get reward from speed
        speed = iface_state.display_speed
        reward += int(speed/20)
        if speed < 5:
            reward -= 1
        elif speed > 50:
            reward += 3
        else:
            reward += 1
        
        # else:
        #     reward -= 1

        # Get reward from getting closer to the finish line
        dist_to_finish_line, distance_to_center_line, angle = get_distance_to_finish_line(car_pos,car_ahead,self.list_points_middle_line,self.road_sections)

        if angle > 0.5:
            reward -= 3
        else:
            reward += 3

        if distance_to_center_line < 6:
            reward += 3
        else:
            reward -= 3

        # if dist_to_finish_line < self.previous_dist_to_finish_line:
        #     reward += 7
        # else:
        #     reward -= 2
        self.previous_dist_to_finish_line = dist_to_finish_line
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
        if _time > 30000:
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
        # if self.is_track_finished:
        #     self.is_track_finished = False

        #     print("Step restarted : Track is finished")
        #     gave_over = True
        #     reward += 10

        score = TRACK_LENGTH - dist_to_finish_line

        return TMSimulationResult(reward, game_over, score)

def get_distance_to_finish_line(car_pos: Point2D, car_ahead: Point2D, list_points_on_middle_line, road_sections):

    # Initialize the minimum distance to a high value
    min_distance = 1000
    min_point = None
    previous_point = None
    
    dist_to_finish_line = 0

    road_section_found = False
    in_section_dist_calculated = False

    # Loop over all the points on the middle line to find the CLOSEST POINT OF THE CENTERLINE TO THE CAR
    for idx, point in enumerate(list_points_on_middle_line[1:]):
        distance = math.sqrt((car_pos.x - point[0])**2 + (car_pos.y - point[1])**2)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if (distance > 1.05 * min_distance) and (min_distance < 12.0):
            break

        if distance < min_distance:
            min_distance = distance
            min_point = point

    idx_point = list_points_on_middle_line.index(min_point)
    if idx_point == 0:
        previous_point = list_points_on_middle_line[1]
    else:
        previous_point = list_points_on_middle_line[idx_point - 1]

    min_point_point2D = Point2D(min_point[0], min_point[1])
    previous_point_point2D = Point2D(previous_point[0], previous_point[1])

    # Distance car to line
    distance_to_line = distance_point_to_line(car_pos, previous_point_point2D, min_point_point2D)

    # Get angle between the car direction and the direction of the track
    angle = angle_between_lines(previous_point_point2D, min_point_point2D, car_pos, car_ahead)

    # Get the road section where the car is located
    for idx, road_section in enumerate(road_sections):

        if road_section_found and in_section_dist_calculated:
            x_section = road_sections[idx]["next_turn_point"][0]
            y_section = road_sections[idx]["next_turn_point"][1]
            x_previous_section = road_sections[idx-1]["next_turn_point"][0]
            y_previous_section = road_sections[idx-1]["next_turn_point"][1]
            dist_to_finish_line += math.sqrt((x_previous_section - x_section)**2 + (y_previous_section - y_section)**2)

        elif road_section_found:
            x_section = road_section["next_turn_point"][0]
            y_section = road_section["next_turn_point"][1]
            dist_to_finish_line = math.sqrt((min_point_point2D.x - x_section)**2 + (min_point_point2D.y - y_section)**2)
            in_section_dist_calculated = True

        elif list(min_point) in road_section["list_middle_points"]:
            road_section_found = True

    # Check if the section is found and if the distance is calculated
            
    # Get the distance to the finish line
    if road_section_found and not in_section_dist_calculated:
        print("road_section_error")
        x_section = road_sections[-1]["next_turn_point"][0]
        y_section = road_sections[-1]["next_turn_point"][1]
        dist_to_finish_line += math.sqrt((min_point_point2D.x - x_section)**2 + (min_point_point2D.y - y_section)**2)

    # If the car is not on a road section, return an error
    elif not road_section_found:
        raise Exception("The road section was not found")

    # print(f"Distance to finish line: {dist_to_finish_line}, Distance to line: {distance_to_line}, Angle: {angle}", flush=True)
    return dist_to_finish_line, distance_to_line, angle
            
# Get the list of points on the middle line of the track
def get_list_point_middle_line():
    # Read the csv file that contains the points on the middle line of the track
    path_to_csv = f'extras/maps/{TRACK_NAME}/road_middle.csv'
    df_points_on_middle_line = pd.read_csv(path_to_csv)
    return list(zip(df_points_on_middle_line.x_values, df_points_on_middle_line.y_values))

def get_road_sections():
    # Read the json file that contains the road sections of the track
    ## Coordinates of turns, straight lines, etc.
    ## Direction of turns, straight lines, etc.
    ## list of points on the middle line of the track related to each road section
    path_to_json = f'extras/maps/{TRACK_NAME}/dict.json'
    with open(path_to_json, "r") as json_file:
        file_data  = json.load(json_file)

    return file_data["road_sections"]

def check_valid_RadarState(state: RadarState) -> bool:
    for detected_point in state.detected_points:
        if detected_point.pos == Point2D(0, 0):
            # print("Invalid RadarState", flush=True)
            return False
        
    return True

def distance_point_to_line(point: Point2D, line_point1: Point2D, line_point2: Point2D) -> float:
    x0, y0 = point.x, point.y
    x1, y1 = line_point1.x, line_point1.y
    x2, y2 = line_point2.x, line_point2.y
    
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return numerator / denominator

def angle_between_lines(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D) -> float:
    # Direction vectors of the lines
    v1 = p2.to_array() - p1.to_array()
    v2 = q2.to_array() - q1.to_array()
    
    # Dot product and norms of the vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Angle in radians
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the value is within the valid range for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta = np.arccos(cos_theta)
    
    return theta


def start_env(databus_buffer: DataBus,
              end_processes,
              tm_speed,
              is_training,
              save_model,
              is_map_render,
              is_curves_render,
              is_tm_speed_changed) -> None:
    print("Environment process started")
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Environment(databus_buffer,
                           end_processes,
                           tm_speed,
                           is_training,
                           save_model,
                           is_map_render,
                           is_curves_render,
                           is_tm_speed_changed))