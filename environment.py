from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

import sys
from collections import namedtuple

from agent import Agent
from model import Model
from experience_buffer import ExperienceBuffer, Experience
from utils import get_distance_to_finish_line, get_list_point_middle_line, get_road_sections

HISTORY_SIZE = 100_000
BATCH_SIZE = 1000
BUFFER_SIZE = 1000

LR = 0.01 #0.001 #0.01
GAMMA = 0.9 #0.95 #0.9

EPSILON_START = 1
EPSILON_END = 0.01 
EPSILON_DECAY = 0.001 #0.00001 #0.001

SYNC_TARGET_EPOCH = 100


class Environment(Client):
    def __init__(self, epsilon, epoch, loss, speed, car_action, time, cancel_training, save_model, end_processes) -> None:
        super(Environment, self).__init__()

        ## Shared memory

        # Training state
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss

        # Car state
        self.speed = speed
        self.car_action = car_action
        self.time = time

        # Actions
        self.cancel_training = cancel_training
        self.save_model = save_model
        self.end_processes = end_processes

        

        # ## To sort out

        # self.experience_buffer = ExperienceBuffer(BUFFER_SIZE)
        # self.model_network = Model(12, 256, 5) #400, 512, 3
        # self.model_target_network = Model(12, 256, 5) #400, 512, 3
        # self.agent = Agent('RL_map_training', self.experience_buffer)
        # self.iter = 0
        # self.inactivity = 0
        # track_name = 'RL_map_training'
        # self.track_name = track_name
        # self.list_point_middle_line = get_list_point_middle_line(track_name)
        # self.road_sections = get_road_sections(track_name)
        # self.previous_dist_to_finish_line = 0
        # self.previous_state = None

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            print(f'Finished the race at {self.race_time}')
            self.finished = True
            # iface.prevent_simulation_finish()
            # iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time >= 0:

            iface_state = iface.get_simulation_state()

            self.speed.value = iface_state.display_speed
            self.time.value = _time

        #     # Get the current state of the car
        #     iface_state = iface.get_simulation_state()
        #     current_state = self.agent.get_state(iface_state)

        #     # Get the action from the agent
        #     action = self.agent.get_action(iface_state, self.model_network, current_state, self.epsilon)

        #     # Get feedback from the action
        #     gave_over = False
        #     reward = 0

        #     # Get reward from speed
        #     speed = iface_state.display_speed
        #     reward += (speed / 30)

        #     # Get reward from getting closer to finish line
        #     dist_to_finish_line = get_distance_to_finish_line(iface_state.position,
        #         self.list_point_middle_line,
        #         self.road_sections
        #    )
        #     if dist_to_finish_line < self.previous_dist_to_finish_line:
        #         reward = 10
        #     else:
        #         reward = -10
        #     self.previous_dist_to_finish_line = dist_to_finish_line

        #     # Restart if too long
        #     if _time > 25000:
        #         gave_over = True
        #         reward = -100
        #         iface.give_up()

        #     # Restart if the car is stopped
        #     if speed < 1:
        #         self.inactivity += 1
        #     else:
        #         self.inactivity = 0

        #     if self.inactivity > 1000:
        #         gave_over = True
        #         reward = -100
        #         iface.give_up()
                

            
            
        #     # Store the experience in the buffer
        #     if self.previous_state is not None:
        #         experience = Experience(self.previous_state, action, reward, current_state)
        #         self.experience_buffer._append(experience)

        #     self.previous_state = current_state
        #     self.iter += 1

        #     if self.iter > 10000:
        #         iface.close()
            
def start_env(epsilon, epoch, loss, speed, car_action, time, cancel_training, save_model, end_processes):
    print('Starting environment...', flush=True)
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...', flush=True)
    run_client(Environment(epsilon, epoch, loss, speed, car_action, time, cancel_training, save_model, end_processes), server_name)