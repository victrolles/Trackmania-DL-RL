import sys
from collections import namedtuple
from typing import Literal

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import torch

from agent import Agent, State
from sac_trainer import SACTrainer
from model import PolicyModel, QModel
from utils import get_distance_to_finish_line, get_list_point_middle_line, get_road_sections

# Input:
INPUT = [
            {  # 0 Up
                "left": False,
                "right": False,
                "accelerate": True,
                "brake": False,
            },
            {  # 1 Left
                "left": True,
                "right": False,
                "accelerate": False,
                "brake": False,
            },
            {  # 2 Right
                "left": False,
                "right": True,
                "accelerate": False,
                "brake": False,
            },
            {  # 3 Up and Left
                "left": True,
                "right": False,
                "accelerate": True,
                "brake": False,
            },
            {  # 4 Up and Right
                "left": False,
                "right": True,
                "accelerate": True,
                "brake": False,
            }
        ]

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done', 'next_state'))

class Environment(Client):
    def __init__(self, episode, policy_loss, q1_loss, q2_loss, best_dist, step, reward, training_time, speed, car_action, game_time, current_dist, is_training_mode, is_model_saved, game_speed, end_processes) -> None:
        super(Environment, self).__init__()

        ## Shared memory

        # Training state
        self.episode = episode
        self.policy_loss = policy_loss
        self.q1_loss = q1_loss
        self.q2_loss = q2_loss
        self.best_dist = best_dist
        self.step = step
        self.reward = reward
        self.training_time = training_time

        # Car state
        self.speed = speed
        self.car_action = car_action
        self.game_time = game_time
        self.current_dist = current_dist

        # Actions
        self.is_training_mode = is_training_mode
        self.is_model_saved = is_model_saved
        self.game_speed = game_speed
        self.end_processes = end_processes

        

        ## To sort out
        track_name = str('RL_map_training')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.game_experience = []

        self.policy_model = PolicyModel(12, 5).to(self.device) #400, 512, 3
        self.q1_model = QModel(12, 5).to(self.device)
        self.q2_model = QModel(12, 5).to(self.device)

        self.agent = Agent(track_name)
        self.dqn_trainer = SACTrainer(self.policy_model, self.q1_model, self.q2_model, self.device, self.is_model_saved, self.end_processes, track_name, self.episode, self.policy_loss, self.q1_loss, self.q2_loss, self.training_time)
        
        self.inactivity = 0
        self.is_track_finished = bool(False)
        self.current_game_speed = 1.0
        
        self.track_name = track_name
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)

        self.previous_dist_to_finish_line = 917.422
        self.previous_state = None
        self.previous_action = None

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)
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

            # ===== Stop the process if needed =====
            if self.end_processes.value:
                # Close the connection to Trackmania
                iface.close()
                # Save the model
                self.dqn_trainer.save_model()
                print("Environment process correctly stopped")
                return
            
            # ===== Save the model if needed =====
            if self.is_model_saved.value:
                self.dqn_trainer.save_model()
                print("Models correctly saved")
                self.is_model_saved.value = False
            
            # ===== Switch between Training and Testing Mode =====
            if not self.is_training_mode.value:
                self.epsilon.value = 0
                return
            
            # ===== Change Training Speed =====
            if self.current_game_speed != self.game_speed.value:
                self.current_game_speed = self.game_speed.value
                iface.set_speed(self.game_speed.value)
            
            # ===== Training =====

            # --- Get REWARD ---

            # Get reward from speed
            iface_state = iface.get_simulation_state()
            reward = 0

            speed = iface_state.display_speed
            reward += int(speed)

            # Get reward from getting closer to finish line
            dist_to_finish_line = get_distance_to_finish_line(iface_state.position,
                self.list_point_middle_line,
                self.road_sections
           )
            if dist_to_finish_line < self.previous_dist_to_finish_line:
                reward += 10 # 70
            else:
                reward -= 10 #70
            self.previous_dist_to_finish_line = dist_to_finish_line

            # Get reward if no lateral contact
            if iface_state.scene_mobil.has_any_lateral_contact:
                reward -= 10 #100
            else:
                reward += 10 #30

            # print(f"Reward : {reward}")

            # --- get DONE ---

            # Restart if too long
            gave_over = False
            if _time > 40000:
                print("Step restarted : Car is too slow")
                gave_over = True
                reward -= 100

            # Restart if the car is stopped
            if speed < 5:
                self.inactivity += 1
            else:
                self.inactivity = 0

            if self.inactivity > 300:

                print("Step restarted : Car is stopped")
                gave_over = True
                reward -= 100

            # Restart if the track is finished
            if self.is_track_finished:
                self.is_track_finished = False

                print("Step restarted : Track is finished")
                gave_over = True
                reward += 100
            
            # --- Get current STATE  ---
            if gave_over:
                current_state = State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                current_state = self.agent.get_state(iface_state, _time)
            
            # ===== Store the experience in the buffer =====
            if self.previous_state is not None and self.previous_action is not None:
                experience = Experience(self.previous_state, self.previous_action, reward, gave_over, current_state)
                self.game_experience.append(experience)

            # ===== Get the current state of the car =====
            if gave_over:
                self.previous_state = None
                self.previous_action = None
                self.car_action.value = -1
            else:
                self.previous_state = current_state
                self.previous_action = self.agent.get_action(self.policy_model, self.previous_state, self.device)
                iface.set_input_state(**INPUT[self.previous_action])
                self.car_action.value = self.previous_action

            # ===== Update the shared memory =====
            self.step.value += 1
            self.reward.value += reward

            self.speed.value = iface_state.display_speed
            self.game_time.value = _time
            self.current_dist.value = 917.422 - dist_to_finish_line
            self.best_dist.value = max(self.best_dist.value, self.current_dist.value)

            # ===== Update the model if game over =====
            if gave_over and len(self.game_experience) > 0:
                self.episode.value += 1
                self.step.value = 0
                self.reward.value = 0
                self.previous_dist_to_finish_line = 917.422
                self.inactivity = 0

                iface.give_up()

                self.dqn_trainer.train_model(self.game_experience)
                self.game_experience = []
                
                iface.give_up()
            
def start_env(episode, policy_loss, q1_loss, q2_loss, best_dist, step, reward, training_time, speed, car_action, game_time, current_dist, is_training_mode, is_model_saved, game_speed, end_processes):
    print("Environment process started")
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Environment(episode, policy_loss, q1_loss, q2_loss, best_dist, step, reward, training_time, speed, car_action, game_time, current_dist, is_training_mode, is_model_saved, game_speed, end_processes))