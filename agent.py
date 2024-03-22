from collections import namedtuple
import time
from typing import Literal

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax
from torch import argmax

from model import PolicyModel
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

    def __init__(self, track_name: str) -> None:

        # Get Track informations
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)

        # Initialize the list of speed and time
        self.list_speed_time = []
    
    def get_state(self, iface_state, game_time) -> State:

        ## Get the speed of the car
        speed = iface_state.display_speed
        # Normalize the speed with factor 0.5 and max speed 300
        speed = np.log(1 + 0.5 * speed) / np.log(1 + 0.5 * 300)

        ## Get the acceleration of the car
        self.list_speed_time.append(Step(speed, game_time))
        if len(self.list_speed_time) < 10:
            acceleration = 1.0
        else:
            old_step = self.list_speed_time.pop(0)  # Remove the oldest step
            acceleration = (speed - old_step.speed) / (game_time - old_step.time)
            acceleration = acceleration * 1e6
        if acceleration == 0.0:
            acceleration = 1.0
        # Normalize the acceleration with factor 1 and max acceleration 20
        acceleration = (acceleration / abs(acceleration)) * np.log(1 + 1* abs(acceleration)) / np.log(1 + 1 * 6000)

        # Get the turning rate of the car
        turning_rate = float(iface_state.scene_mobil.turning_rate)

        # Get the is_free_wheeling of the car
        is_free_wheeling = float(iface_state.scene_mobil.is_freewheeling)

        # Get the is_sliding of the car
        is_sliding = float(iface_state.scene_mobil.is_sliding)

        # Get the has_any_lateral_contact of the car
        has_any_lateral_contact = float(iface_state.scene_mobil.has_any_lateral_contact)

        # Get other positional informations
        list_infos = get_positional_informations(
            iface_state.position,
            iface_state.yaw_pitch_roll[0],
            self.list_point_middle_line,
            self.road_sections
            )
        
        # Get the distance and angle to the centerline
        distance_to_centerline = list_infos[0] / 12.0
        angle_to_centerline = list_infos[1] / np.pi

        # Get the distance to the next turns
        distance_to_first_turn = list_infos[2]
        distance_to_first_turn = np.log(1 + 2 * distance_to_first_turn) / np.log(1 + 2 * 500)
        distance_to_second_turn = list_infos[3]
        distance_to_second_turn = np.log(1 + 2 * distance_to_second_turn) / np.log(1 + 2 * 500)

        # Get the direction of the next turns
        direction_of_first_turn = list_infos[4]
        direction_of_second_turn = list_infos[5]

        # print(f"Spd: {speed:.2f}, Acc: {acceleration:.2f}, TR: {turning_rate:.2f}, Ctt: {has_any_lateral_contact:.2f}, Dist cen: {distance_to_centerline:.2f}, Angl cen: {angle_to_centerline:.2f}, Dist1t: {distance_to_first_turn:.2f}, Dist2t: {distance_to_second_turn:.2f}, Dir1t: {direction_of_first_turn:.2f}, Dir2t: {direction_of_second_turn:.2f}")

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
    
    def get_action(self,
                   policy_model: PolicyModel,
                   state: State,
                   device: Literal['cpu', 'cuda'],
                   deterministic: bool = False) -> int:
        """
        Generate actions from the policy model given the current state.

        Parameters:
        - policy_model: The instantiated PolicyModel used by the SAC agent.
        - state: The current state of the environment, expected to be a tensor.
        - deterministic: A boolean indicating whether to use the mean of the action distribution as the action
        (True, deterministic) or to sample from the distribution (False, stochastic).

        Returns:
        - action: The action to be taken, as a numpy array.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Convert state to tensor and add batch dimension
        action_probabilities = policy_model(state_tensor)
        probabilities = softmax(action_probabilities, dim=1)

        if deterministic:
            # Select the action with the highest probability
            action = argmax(probabilities, dim=1)
        else:
            # Sample an action according to the probabilities for exploration
            action_distribution = Categorical(probabilities)
            action = action_distribution.sample()

        action = action.item()  # Convert to Python int
        return action