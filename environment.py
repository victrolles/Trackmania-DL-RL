from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

import sys
from collections import namedtuple

from agent import Agent
from dqn_trainer import DQNTrainer
from model import Model
from experience_buffer import ExperienceBuffer, Experience
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

class Environment(Client):
    def __init__(self, epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, cancel_training, save_model, end_processes) -> None:
        super(Environment, self).__init__()

        ## Shared memory

        # Training state
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss
        self.best_dist = best_dist
        self.current_dist = current_dist
        self.buffer_size = buffer_size

        # Car state
        self.speed = speed
        self.car_action = car_action
        self.time = time

        # Actions
        self.cancel_training = cancel_training
        self.save_model = save_model
        self.end_processes = end_processes

        

        ## To sort out

        self.experience_buffer = ExperienceBuffer()
        self.model_network = Model(12, 256, 5) #400, 512, 3
        self.model_target_network = Model(12, 256, 5) #400, 512, 3
        self.agent = Agent('RL_map_training', self.experience_buffer)
        self.dqn_trainer = DQNTrainer(self.model_network, self.model_target_network, self.experience_buffer, epsilon, epoch, loss)
        self.inactivity = 0
        self.test_model = False
        track_name = 'RL_map_training'
        self.track_name = track_name
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)

        self.previous_dist_to_finish_line = 0
        self.previous_state = None
        self.previous_action = None

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)
        iface.give_up()

    # Function to detect if the car crossed the finish line
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()
            iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time >= 0:

            # ===== Stop the process if needed =====
            if self.end_processes.value:
                iface.close()
                print("Environment process correctly stopped", flush=True)
                return
            
            # ===== Switch between Training and Testing Mode =====
            if self.cancel_training.value:
                iface.give_up()
                self.epsilon.value = 0
                return
            
            # ===== Training =====

            # --- Get REWARD ---

            # Get reward from speed
            iface_state = iface.get_simulation_state()
            reward = 0
            speed = iface_state.display_speed
            reward += (speed / 30)

            # Get reward from getting closer to finish line
            dist_to_finish_line = get_distance_to_finish_line(iface_state.position,
                self.list_point_middle_line,
                self.road_sections
           )
            if dist_to_finish_line < self.previous_dist_to_finish_line:
                reward = 10
            else:
                reward = -10
            self.previous_dist_to_finish_line = dist_to_finish_line

            # --- get DONE ---

            # Restart if too long
            gave_over = False
            if _time > 25000:
                print("Too long", flush=True)
                gave_over = True
                reward += -100

            # Restart if the car is stopped
            if speed < 5:
                self.inactivity += 1
            else:
                self.inactivity = 0

            if self.inactivity > 1000:
                print("Car is stopped for too long", flush=True)
                self.inactivity = 0
                gave_over = True
                reward += -100
            
            # --- Get STATE ---
            current_state = self.agent.get_state(iface_state)
            
            # ===== Store the experience in the buffer =====
            if self.previous_state is not None and self.previous_action is not None:
                experience = Experience(self.previous_state, self.previous_action, reward, gave_over, current_state)
                self.experience_buffer._append(experience)
                self.buffer_size.value = len(self.experience_buffer)

            # ===== Get the current state of the car =====
            self.previous_state = self.agent.get_state(iface_state)
            self.previous_action = self.agent.get_action(self.model_network, self.previous_state, self.epsilon)
            iface.set_input_state(**INPUT[self.previous_action])
            self.car_action.value = self.previous_action

            # ===== Update the shared memory =====
            self.speed.value = iface_state.display_speed
            self.time.value = _time
            self.current_dist.value = 900.0 - dist_to_finish_line
            self.best_dist.value = max(self.best_dist.value, self.current_dist.value)

            # ===== Update the model if game over =====
            if gave_over:
                iface.give_up()
                self.dqn_trainer.train_model()
                iface.give_up()
            
def start_env(epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, cancel_training, save_model, end_processes):
    print("Environment process started", flush=True)
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...', flush=True)
    run_client(Environment(epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, cancel_training, save_model, end_processes), server_name)