import sys
import math
import json

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import pandas as pd
import torch

from config.data_classes import Point2D, DataBus, TMSimulationResult, Exp
from config.globals import TRACK_NAME, TRACK_LENGTH, BUFFER_SIZE
from config.dictionaries import INPUT

from tm_agents.radar_agent import RadarAgent

from rl_algorithms.dqn.dqn_trainer import DQNTrainer
from rl_algorithms.experience_buffer import ExperienceBuffer

class Environment(Client):
    def __init__(self, databus_buffer: DataBus, end_processes) -> None:
        super(Environment, self).__init__()

        ## Shared memory
        self.databus_buffer = databus_buffer
        self.end_processes = end_processes


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iter = 0
        self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        self.previous_dist_to_finish_line = TRACK_LENGTH
        self.previous_state = None
        self.previous_action = None
        self.inactivity = 0

        
        list_points_left_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_left.csv')
        list_points_right_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_right.csv')
        list_points_left_border = list_points_left_border.iloc[::25]
        list_points_right_border = list_points_right_border.iloc[::25]
        self.list_points_left_border = list(zip(list_points_left_border.x_values, list_points_left_border.y_values))
        self.list_points_right_border = list(zip(list_points_right_border.x_values, list_points_right_border.y_values))
        print("size left: ", len(self.list_points_left_border))
        print("size right: ", len(self.list_points_right_border))
        self.list_points_middle_line = get_list_point_middle_line()
        self.road_sections = get_road_sections()

        
        self.agent = RadarAgent(self.list_points_left_border, self.list_points_right_border)
        self.dqn_trainer = DQNTrainer(self.experience_buffer, self.device)

        # ax.set(xlabel='x', ylabel='y')
        # ax.legend()
        # plt.show()

        # # Setup screen and game
        # def set_window_pos(window_name, x, y, width, height):
        #     hwnd = win32gui.FindWindow(None, window_name)
        #     if hwnd:
        #         # print(win32gui.GetWindowRect(hwnd))
        #         win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, width, height, 0)

        # # Example usage
        # name = "TrackMania Nations Forever (TMInterface 1.4.3)"
        # set_window_pos(name, -6, 0, 256, 256)  # Replace 'Google Chrome' with the exact window title

        # ## To sort out
        # track_name = str('snake_map_training')
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.game_experience = []

        # self.policy_model = PolicyModel(12, 5).to(self.device) #400, 512, 3
        # self.q1_model = QModel(12, 5).to(self.device)
        # self.q2_model = QModel(12, 5).to(self.device)

        # self.agent = Agent(track_name)
        # self.dqn_trainer = SACTrainer(self.policy_model, self.q1_model, self.q2_model, self.device, self.is_model_saved, self.end_processes, track_name, self.episode, self.policy_loss, self.q1_loss, self.q2_loss, self.training_time)
        # self.dqn_trainer = DQNTrainer(self.game_experience, self.epsilon, self.epoch, self.loss, self.device, self.is_model_saved, self.end_processes, track_name, self.training_time)
        # self.inactivity = 0
        # self.is_track_finished = bool(False)
        # self.current_game_speed = 1.0
        
        # self.track_name = track_name
        # self.list_point_middle_line = get_list_point_middle_line(track_name)
        # self.road_sections = get_road_sections(track_name)

        

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)
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

            if self.iter % 20 == 0:

                # ===== Stop the process if needed =====
                if self.end_processes.value:
                    self.stop_env_process(iface)

                iface_state = iface.get_simulation_state()
                
                # ===== Get the state =====
                state = self.agent.get_state(iface_state)
                # print(state.distances, flush=True)

                # ===== Get the TM simulation result =====
                tm_simulation_result = self.get_TM_simulation_result(iface_state, _time)

                # ===== Store experience =====
                if self.previous_state is not None and self.previous_action is not None:
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
                    self.previous_action = self.agent.get_action(self.dqn_trainer.model_network, self.previous_state.distances, self.dqn_trainer.epsilon, self.device)
                    # print(self.previous_action, flush=True)
                    iface.set_input_state(**INPUT[self.previous_action])

                data_bus = DataBus(state.car_pos, state.car_ahead_pos, state.detected_points)
                self.databus_buffer.put(data_bus)

                # ===== Update the model if game over =====
                if tm_simulation_result.done and len(self.experience_buffer) > 0:
                    self.previous_dist_to_finish_line = TRACK_LENGTH
                    self.inactivity = 0

                    iface.give_up()

                    self.dqn_trainer.train_model()
                    
                    iface.give_up()

    def stop_env_process(self, iface: TMInterface) -> None:
        # Close the connection to Trackmania
        iface.close()
        # Save the model
        self.dqn_trainer.save_model()
        print("Environment process correctly stopped", flush=True)
        return     

    # get reward, done, score        
    def get_TM_simulation_result(self, iface_state: TMInterface, _time: int) -> TMSimulationResult:
        
        reward = 0
        game_over = False
        
        # Get reward from speed
        speed = iface_state.display_speed
        if speed > 60:
            reward += 1
        else:
            reward -= 1

        # Get reward from getting closer to the finish line
        dist_to_finish_line = get_distance_to_finish_line(iface_state.position,
                                                          self.list_points_middle_line,
                                                          self.road_sections)
        if dist_to_finish_line < self.previous_dist_to_finish_line:
            reward += 2
        else:
            reward -= 2
        self.previous_dist_to_finish_line = dist_to_finish_line

        # Get reward from getting closer to the middle line
        #TODO: Get the distance to the middle line

        # Get reward if no lateral contact
        if iface_state.scene_mobil.has_any_lateral_contact:
            reward -= 1
        else:
            reward += 1

        # --- get DONE ---

        # Restart if too long
        if _time > 30000:
            print("Step restarted : Car is too slow", flush=True)
            game_over = True
            reward -= 10

        # Restart if the car is stopped
        if speed < 5:
            self.inactivity += 1
        else:
            self.inactivity = 0
        # print(self.inactivity, flush=True)

        if self.inactivity > 20:
            print("Step restarted : Car is stopped", flush=True)
            game_over = True
            reward -= 10

        # Restart if the track is finished
        # if self.is_track_finished:
        #     self.is_track_finished = False

        #     print("Step restarted : Track is finished")
        #     gave_over = True
        #     reward += 10

        score = TRACK_LENGTH - dist_to_finish_line

        return TMSimulationResult(reward, game_over, score)

def get_distance_to_finish_line(car_location, list_points_on_middle_line, road_sections):

    # Initialize the minimum distance to a high value
    min_distance = 1000
    min_point = None
    
    dist_to_finish_line = 0

    road_section_found = False
    in_section_dist_calculated = False

    # Loop over all the points on the middle line to find the CLOSEST POINT OF THE CENTERLINE TO THE CAR
    for point in list_points_on_middle_line[1:]:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if (distance > 1.05 * min_distance) and (min_distance < 12.0):
            break

        if distance < min_distance:
            min_distance = distance
            min_point = point

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
            dist_to_finish_line = math.sqrt((min_point[0] - x_section)**2 + (min_point[1] - y_section)**2)
            in_section_dist_calculated = True

        elif list(min_point) in road_section["list_middle_points"]:
            road_section_found = True

    # Check if the section is found and if the distance is calculated
            
    # Get the distance to the finish line
    if road_section_found and not in_section_dist_calculated:
        print("road_section_error")
        x_section = road_sections[-1]["next_turn_point"][0]
        y_section = road_sections[-1]["next_turn_point"][1]
        dist_to_finish_line += math.sqrt((min_point[0] - x_section)**2 + (min_point[1] - y_section)**2)

    # If the car is not on a road section, return an error
    elif not road_section_found:
        raise Exception("The road section was not found")

    return dist_to_finish_line    
            
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


def start_env(databus_buffer: DataBus, end_processes) -> None:
    print("Environment process started")
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Environment(databus_buffer, end_processes))