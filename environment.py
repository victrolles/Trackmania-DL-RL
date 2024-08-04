import sys
import time

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import torch
import numpy as np

from librairies.data_classes import Point2D, DataBus, Exp
from librairies.globals import TRACK_LENGTH, BUFFER_SIZE, FRAME_COOLDOWN_ACTION, FRAME_COOLDOWN_DISPLAY_STATE, FRAME_COOLDOWN_DISPLAY_STATS, INITIAL_GAME_SPEED, RANDOM_SPAWN, SPAWN_NUMBER, SAVE_MODELS_RATE
from librairies.dictionaries import INPUT, Rd
from librairies.tm_math_functions import get_road_points, get_closest_point, distance_to_finish_line

from tm_agents.radar_agent import RadarAgent

from rl_algorithms.dqn.dqn_trainer import DQNTrainer
from rl_algorithms.experience_buffer import ExperienceBuffer

from tm_rewards.simulation_rewards import SimulationRewards

class Environment(Client):
    def __init__(self,
                 databus_buffer: DataBus,
                 end_processes,
                 tm_speed,
                 is_training,
                 saving_model,
                 is_map_render,
                 is_curves_render,
                 is_tm_speed_changed,
                 is_random_spawn) -> None:
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
        self.is_random_spawn = is_random_spawn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iter = 0
        
        self.dist_to_finish_line = 0
        self.start_dist_to_finish_line = 0
        self.previous_state = None
        self.previous_action = None
        self.previous_tm_simulation_result = None
        self.start_simulation_time = time.time()
        self.has_respawned = False
        self.has_get_start_dist = False
        self.is_track_finished = False
        self.current_game_speed = INITIAL_GAME_SPEED
        self.is_random_spawn.value = RANDOM_SPAWN
        self.training_stats = None
        self.middle_points = get_road_points(Rd.MIDDLE, True)
        self.start_total_time = time.time()
        

        
        # list_points_left_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_left.csv')
        # list_points_right_border = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_right.csv')
        # list_points_left_border = list_points_left_border.iloc[::20]
        # list_points_right_border = list_points_right_border.iloc[::20]
        # self.list_points_left_border = list(zip(list_points_left_border.x_values, list_points_left_border.y_values))
        # self.list_points_right_border = list(zip(list_points_right_border.x_values, list_points_right_border.y_values))
        # print("size left: ", len(self.list_points_left_border))
        # print("size right: ", len(self.list_points_right_border))
        # self.list_points_middle_line = get_list_point_middle_line()
        # self.road_sections = get_road_sections()

        self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        self.agent = RadarAgent()
        self.dqn_trainer = DQNTrainer(self.experience_buffer, self.device)
        self.tm_simulation = SimulationRewards(self.is_track_finished)
        

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)

        iface.set_speed(self.current_game_speed)
        self.tm_speed.value = self.current_game_speed

        iface.set_timeout(20_000)
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

            if self.has_respawned and _time > 100 and _time < 1000 :
                self.has_respawned = False
                random_int = np.random.randint(0, SPAWN_NUMBER)
                string = f"load_state checkpoint_{random_int}.bin"
                iface.execute_command(string)
                

            if not self.has_respawned and self.has_get_start_dist and _time > 300 :
                self.has_get_start_dist = False
                iface_state = iface.get_simulation_state()
                car_pos = Point2D(iface_state.position[0], iface_state.position[2])
                self.start_dist_to_finish_line = distance_to_finish_line(car_pos, get_closest_point(car_pos, self.middle_points), self.middle_points)

            if self.iter % FRAME_COOLDOWN_ACTION == 0:

                # ===== Stop the process if needed =====
                if self.end_processes.value:
                    self.stop_env_process(iface)

                # ===== Change Training Speed =====
                if self.is_tm_speed_changed.value:
                    self.change_tm_speed(iface)

                # ===== Save the model if needed =====
                if self.saving_model.value:
                    if self.iter % SAVE_MODELS_RATE == 0:
                        self.save_model()

                iface_state = iface.get_simulation_state()
                
                # ===== Get the state =====
                state = self.agent.get_state(iface_state)
                # print(state.distances, flush=True)

                # ===== Get the TM simulation result =====
                tm_simulation_result = self.tm_simulation.get_TM_simulation_result(iface_state, state.car_pos, state.car_ahead_pos, self.start_simulation_time)
                self.dist_to_finish_line = tm_simulation_result.dist_to_finish_line

                # print(f"Reward: {tm_simulation_result.reward}", flush=True)

                # ===== Store experience =====
                if self.previous_state is not None and self.previous_action is not None:
                    if self.agent.check_validity(state):
                        experience = Exp(self.previous_state.distances,
                                        self.previous_action,
                                        self.previous_tm_simulation_result.reward,
                                        self.previous_tm_simulation_result.done,
                                        state.distances)
                        self.experience_buffer._append(experience)

                # ===== Get the current state of the car =====
                if tm_simulation_result.done:
                    self.previous_state = None
                    self.previous_action = None
                    self.previous_tm_simulation_result = None
                else:
                    self.previous_state = state
                    self.previous_tm_simulation_result = tm_simulation_result
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

                    if not RANDOM_SPAWN:
                        self.start_dist_to_finish_line = TRACK_LENGTH

                    dist = self.start_dist_to_finish_line - self.dist_to_finish_line

                    if self.iter % FRAME_COOLDOWN_DISPLAY_STATS == 0 and self.is_curves_render.value:
                        self.databus_buffer.put(DataBus(None,
                                                        self.training_stats,
                                                        time.time() - self.start_total_time,
                                                        dist,
                                                        self.iter/(time.time() - self.start_total_time),
                                                        len(self.experience_buffer)))
                    
                    iface.give_up()

                    self.start_simulation_time = time.time()

                    if self.is_random_spawn.value:
                        self.has_respawned = True
                        self.has_get_start_dist = True

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
        self.is_tm_speed_changed.value = False

    def save_model(self) -> None:
        self.dqn_trainer.save_model()
        print("Models correctly saved", flush=True)
        self.saving_model.value = False


def start_env(databus_buffer: DataBus,
              end_processes,
              tm_speed,
              is_training,
              save_model,
              is_map_render,
              is_curves_render,
              is_tm_speed_changed,
              is_random_spawn) -> None:
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
                           is_tm_speed_changed,
                           is_random_spawn))