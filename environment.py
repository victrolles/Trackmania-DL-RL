import sys
import time

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import torch

from librairies.data_classes import DataBus, Exp
from librairies.dictionaries import INPUT, Rd
from librairies.tm_math_functions import get_road_points
from librairies.timer import Timer
from librairies.tm_respawn import TMRandomRespawn

from tm_agents.radar_agent import RadarAgent

from rl_algorithms.dqn.dqn_trainer import DQNTrainer
from rl_algorithms.experience_buffer import ExperienceBuffer

from tm_rewards.simulation_rewards import SimulationRewards

from config import Config

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

        ## Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ## Config
        self.config = Config()
        self.tm_speed.value = self.config.environment.game_speed
        self.is_training.value = self.config.traininig
        self.is_random_spawn.value = self.config.spawn_config.random

        ## Road points
        self.middle_points = get_road_points(Rd.MIDDLE, True)

        ## TM Data
        self.iter = 0
        self.start_dist_to_finish_line = 0
        self.is_track_finished = False
        self.tm_rr = TMRandomRespawn(self.config.spawn_config, self.middle_points)

        ## Experience from previous state
        self.previous_exp = Exp.set_none()

        ## Timers
        self.simulation_timer = Timer("Simulation")
        self.timer = Timer("Global")
        self.training_timer = Timer("Training")

        self.timer.start()
        self.simulation_timer.start()
        self.training_timer.start()
        self.training_timer.pause()

        ## Reinforcement Learning Classes
        self.experience_buffer = ExperienceBuffer(self.config.exp_buffer_config.buffer_size)
        self.agent = RadarAgent(self.config)
        self.dqn_trainer = DQNTrainer(self.config.rl_config, self.experience_buffer, self.agent.input_size, self.device)
        self.tm_simulation = SimulationRewards(self.is_track_finished)
        

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)

        iface.set_speed(self.tm_speed.value)

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

            # ---- Random Respawn ----
            self.tm_rr.respawn(iface, _time)
            
            # ---- Simulation ----
            if self.iter % self.config.cooldown_config.action == 0:

                # ===== Stop the process if needed =====
                if self.end_processes.value:
                    self.stop_env_process(iface)

                # ===== Change Training Speed =====
                if self.is_tm_speed_changed.value:
                    self.change_tm_speed(iface)

                # ===== Save the model if needed =====
                if self.saving_model.value or self.iter % self.config.rl_config.sync_save_rate == 0:
                    self.save_model()

                iface_state = iface.get_simulation_state()
                
                # ===== Get the state =====
                state = self.agent.get_state(iface_state)
                # print(state.distances, flush=True)

                # ===== Get the TM simulation result =====
                tm_simulation_result = self.tm_simulation.get_TM_simulation_result(iface_state, state.car_pos, state.car_ahead_pos, self.start_simulation_time)
                dist_to_finish_line = tm_simulation_result.dist_to_finish_line

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

                if self.iter % self.config.cooldown_config.display_state == 0 and self.is_map_render.value:
                    self.databus_buffer.put(DataBus(state,
                                                    None,
                                                    self.timer.get_time(),
                                                    0,
                                                    self.iter/self.timer.get_time(),
                                                    len(self.experience_buffer)))

                # ===== Update the model if game over =====
                if tm_simulation_result.done and len(self.experience_buffer) > 0:
                    self.simulation_timer.pause()
                    iface.give_up()

                    self.training_timer.resume()
                    training_stats = self.dqn_trainer.train_model()
                    self.training_timer.pause()

                    if not self.is_random_spawn.value:
                        self.start_dist_to_finish_line = self.config.environment.length
                    else:
                        self.tm_rr.set_random_spawn()

                    dist = self.start_dist_to_finish_line - dist_to_finish_line

                    if self.iter % self.config.cooldown_config.display_stats == 0 and self.is_curves_render.value:
                        self.databus_buffer.put(DataBus(None,
                                                        training_stats,
                                                        self.timer.get_time(),
                                                        dist,
                                                        self.iter/self.timer.get_time(),
                                                        len(self.experience_buffer)))
                    
                    iface.give_up()
                    self.simulation_timer.resume()
                        

    def stop_env_process(self, iface: TMInterface) -> None:
        # Close the connection to Trackmania
        iface.close()
        # Save the model
        self.dqn_trainer.save_model()
        print("Environment process correctly stopped", flush=True)
        exit()

    def change_tm_speed(self, iface: TMInterface) -> None:
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