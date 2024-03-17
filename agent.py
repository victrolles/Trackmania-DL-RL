from collections import namedtuple
import time
import numpy as np
import torch

from utils import get_list_point_middle_line, get_road_sections, get_positional_informations



# The State class is a named tuple that contains all the information that the agent needs to take a decision
State = namedtuple('State',
    (
        'speed',
        'acceleration',
        'turning_rate',
        'is_free_wheeling',
        'is_sliding',
        'has_any_lateral_contact',
        'distance_to_centerline',
        'angle_to_centerline',
        'distance_to_first_turn',
        'distance_to_second_turn',
        'direction_of_first_turn',
        'direction_of_second_turn'
    ))
Step = namedtuple('Step',
    (
        'speed',
        'time'
    ))

class Agent:

    def __init__(self, track_name: str, experience_buffer):
        self.track_name = track_name
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)

        # Initialize the experience buffer
        self.experience_buffer = experience_buffer

        # Initialize the list of speed and time
        self.list_speed_time = []
    
    def get_state(self, iface_state) -> State:

        # Get the speed of the car
        speed = iface_state.display_speed

        # Get the acceleration of the car
        self.list_speed_time.append(Step(speed, time.time()))
        if len(self.list_speed_time) < 10:
            acceleration = 0
        else:
            old_step = self.list_speed_time.pop(0)  # Remove the oldest step
            acceleration = (speed - old_step.speed) / (time.time() - old_step.time)

        # Get the turning rate of the car
        turning_rate = iface_state.scene_mobil.turning_rate

        # Get the is_free_wheeling of the car
        is_free_wheeling = iface_state.scene_mobil.is_freewheeling

        # Get the is_sliding of the car
        is_sliding = iface_state.scene_mobil.is_sliding

        # Get the has_any_lateral_contact of the car
        has_any_lateral_contact = iface_state.scene_mobil.has_any_lateral_contact

        # Get other positional informations
        list_infos = get_positional_informations(
            iface_state.position,
            iface_state.yaw_pitch_roll[0],
            self.list_point_middle_line,
            self.road_sections
            )
        
        # Get the distance and angle to the centerline
        distance_to_centerline = list_infos[0]
        angle_to_centerline = list_infos[1]

        # Get the distance to the next turns
        distance_to_first_turn = list_infos[2]
        distance_to_second_turn = list_infos[3]

        # Get the direction of the next turns
        direction_of_first_turn = list_infos[4]
        direction_of_second_turn = list_infos[5]

        return State(
            speed,
            acceleration,
            turning_rate,
            is_free_wheeling,
            is_sliding,
            has_any_lateral_contact,
            distance_to_centerline,
            angle_to_centerline,
            distance_to_first_turn,
            distance_to_second_turn,
            direction_of_first_turn,
            direction_of_second_turn
        )
    
    def get_action(self, model_network, state, epsilon, device):
        # Espilon-Greedy: tradeoff exploration / exploitation
        if np.random.random() < epsilon.value:
            move = np.random.randint(0, 5)
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = model_network(state0)
            move = torch.argmax(prediction).item()

        return move