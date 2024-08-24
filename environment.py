import sys
import os
import datetime

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import torch

from librairies.data_classes import DataBus, TimeStats
from librairies.dictionaries import INPUT, Rd
from librairies.tm_math_functions import get_road_points
from librairies.timers import Timers
from librairies.tm_respawn import TMRandomRespawn
from librairies.scores import Scores
from librairies.exp import Exp

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

        ## ---- Bus data ----
        self.databus_buffer = databus_buffer

        ## ---- Shared memory ----
        self.end_processes = end_processes
        self.tm_speed = tm_speed
        self.is_training = is_training
        self.saving_model = saving_model
        self.is_map_render = is_map_render
        self.is_curves_render = is_curves_render
        self.is_tm_speed_changed = is_tm_speed_changed
        self.is_random_spawn = is_random_spawn

        ## ---- Device ----
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ## ---- Config ----
        self.config = Config()
        if self.config.rl_config.load_checkpoint:
            self.checkpoint = torch.load(self.config.rl_config.load_checkpoint_path)
            # self.config = self.checkpoint['config']

        self.tm_speed.value = self.config.environment.game_speed
        self.is_training.value = self.config.rl_config.traininig
        self.is_random_spawn.value = self.config.spawn_config.random

        ## ---- Road points ----
        self.middle_points = get_road_points(self.config.environment.name, Rd.MIDDLE, True)

        ## ---- TM Data ----
        self.iter = 0
        self.save_dir = ''  
        self.start_dist_to_finish_line = 0
        self.is_track_finished = False
        self.tm_rr = TMRandomRespawn(self.config.spawn_config, self.middle_points)

        ## ---- Experience from previous state ----
        self.previous_exp = Exp(None, None, None, None, None)

        ## ---- Timers ----
        if self.config.rl_config.load_checkpoint:
            self.timers = self.checkpoint['timers']
        else:
            self.timers = Timers()

            self.timers.add_timer("Simulation")
            self.timers.add_timer("Global")
            self.timers.add_timer("Training")

            self.timers.start("Global")
            self.timers.start("Training")
            self.timers.pause("Training")

        self.timers.start("Simulation")

        ## ---- Score ----
        if self.config.rl_config.load_checkpoint:
            self.scores = self.checkpoint['scores']
        else:
            self.scores = Scores()

        ## ---- Send datas to the bus ----
        self.databus_buffer.put(DataBus(self.config.spawn_config,
                                        self.scores,
                                        None,
                                        None,
                                        self.get_timeStats(),
                                        None,
                                        None,
                                        None))

        ## ---- Reinforcement Learning Classes ----
        self.experience_buffer = ExperienceBuffer(self.config.exp_buffer_config.buffer_size)
        self.agent = RadarAgent(self.config)
        self.tm_simulation = SimulationRewards(self.config.environment.name, self.is_track_finished)

        self.dqn_trainer = DQNTrainer(self.config.rl_config, self.experience_buffer, self.agent.input_size, self.device)
        if self.config.rl_config.load_checkpoint:
            self.dqn_trainer.load_model(self.checkpoint)
        

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
                if self.saving_model.value or (self.dqn_trainer.epoch % self.config.rl_config.sync_save_rate == 0 and self.dqn_trainer.epoch != self.dqn_trainer.prev_epoch):
                    self.make_checkpoint()

                iface_state = iface.get_simulation_state()
                
                # ===== Get the state =====
                state = self.agent.get_state(iface_state)
                # print(state.distances, flush=True)

                # ===== Get the TM simulation result =====
                tm_simulation_result = self.tm_simulation.get_TM_simulation_result(iface_state, state.car_pos, state.car_ahead_pos, self.timers.get_time("Simulation"))
                dist_to_finish_line = tm_simulation_result.dist_to_finish_line
                # print(f"Distance to finish line: {dist_to_finish_line}", flush=True)

                # print(f"Reward: {tm_simulation_result.reward}", flush=True)

                # ===== Store experience =====
                if not self.previous_exp.is_none():
                    if self.agent.check_validity(state):
                        experience = Exp(self.previous_exp.state.distances,
                                        self.previous_exp.action,
                                        self.previous_exp.reward,
                                        self.previous_exp.done,
                                        state.distances)
                        self.experience_buffer._append(experience)

                # ===== Get the current state of the car =====
                if tm_simulation_result.done:
                    self.previous_exp.set_none()
                else:
                    self.previous_exp.state = state
                    self.previous_exp.reward = tm_simulation_result.reward
                    self.previous_exp.done = tm_simulation_result.done
                    if iface_state.position[2] < 122:
                        self.previous_exp.action = 0
                    else:
                        self.previous_exp.action = self.agent.get_action(self.dqn_trainer.model_network,
                                                                    self.previous_exp.state.distances,
                                                                    self.dqn_trainer.epsilon,
                                                                    self.device,
                                                                    self.is_training.value)
                    # print(self.previous_action, flush=True)
                    iface.set_input_state(**INPUT[self.previous_exp.action])

                if self.iter % self.config.cooldown_config.display_state == 0:
                    self.databus_buffer.put(DataBus(None,
                                                    None,
                                                    state,
                                                    None,
                                                    self.get_timeStats(),
                                                    0,
                                                    self.iter/self.timers.get_time("Global"),
                                                    len(self.experience_buffer)))

                # ===== Update the model if game over =====
                if tm_simulation_result.done and len(self.experience_buffer) > 0:
                    self.timers.stop("Simulation")
                    iface.give_up()

                    self.timers.resume("Training")
                    training_stats = self.dqn_trainer.train_model()
                    self.timers.pause("Training")

                    if not self.is_random_spawn.value:
                        self.start_dist_to_finish_line = self.config.environment.length
                    else:
                        self.tm_rr.set_random_spawn()
                        self.start_dist_to_finish_line = self.tm_rr.start_dist_to_finish_line

                    dist = self.start_dist_to_finish_line - dist_to_finish_line
                    # print(f"Distance to finish line: {dist}", flush=True)
                    self.scores.add_score(dist)

                    if self.iter % self.config.cooldown_config.display_stats == 0:
                        self.databus_buffer.put(DataBus(None,
                                                        None,
                                                        None,
                                                        training_stats,
                                                        self.get_timeStats(),
                                                        dist,
                                                        self.iter/self.timers.get_time("Global"),
                                                        len(self.experience_buffer)))
                    
                    iface.give_up()
                    self.timers.start("Simulation")
                        

    def stop_env_process(self, iface: TMInterface) -> None:
        # Close the connection to Trackmania
        iface.close()
        # Save the model
        self.make_checkpoint()
        print("Environment process correctly stopped", flush=True)
        exit()

    def change_tm_speed(self, iface: TMInterface) -> None:
        iface.set_speed(self.tm_speed.value)
        self.is_tm_speed_changed.value = False

    def get_timeStats(self) -> TimeStats:
        return TimeStats(self.timers.get_time("Global"),
                        self.timers.get_time("Training"),
                        self.timers.get_time("Simulation"))

    def make_checkpoint(self):
        # Create the save directory
        if self.save_dir == '':
            self.save_dir = f"extras/maps/{self.config.environment.name}/saves/{datetime.datetime.now().strftime("%d-%m-%y-%H-%M")}_{self.dqn_trainer.name}_{self.config.agent_config.name}Agent"
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"Created directory: {self.save_dir}")

        save_path = os.path.join(self.save_dir, f"model_{self.dqn_trainer.epoch}.pth")

        dict_env = {
            # config
            'config': self.config,

            # training stats
            'timers': self.timers,
            'scores': self.scores
            }
        
        dict_rl = self.dqn_trainer.save_model()

        torch.save({**dict_env, **dict_rl}, save_path)

        print(f"Models correctly saved at epoch {self.dqn_trainer.epoch}")
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