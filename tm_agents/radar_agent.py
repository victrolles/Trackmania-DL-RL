import numpy as np
import torch

from librairies.data_classes import RadarState, Point2D
from librairies.tm_math_functions import point_ahead, closest_intersection, get_road_points
from librairies.dictionaries import Rd

class RadarAgent:

    def __init__(self) -> None:

        self.input_nbr = 10
        
        self.left_border_points = get_road_points(Rd.LEFT, True)
        self.right_border_points = get_road_points(Rd.RIGHT, True)

    def get_state(self, iface_state) -> RadarState:
        
        car_pos = Point2D(iface_state.position[0], iface_state.position[2])
        car_pos_ahead = point_ahead(car_pos, iface_state.yaw_pitch_roll[0])

        speed = iface_state.display_speed
        normalized_speed = min(speed, 100) / 100

        detected_point = closest_intersection(car_pos, car_pos_ahead, self.left_border_points, self.right_border_points)
        detected_points = [detected_point]
        distances = [detected_point.dist]

        for i in range(1, 5):
            car_pos_ahead1 = point_ahead(car_pos, iface_state.yaw_pitch_roll[0], np.pi/11*i)
            detected_point = closest_intersection(car_pos, car_pos_ahead1, self.left_border_points, self.right_border_points)
            detected_points.append(detected_point)
            distances.append(detected_point.dist)

            car_pos_ahead2 = point_ahead(car_pos, iface_state.yaw_pitch_roll[0], -np.pi/11*i)
            detected_point = closest_intersection(car_pos, car_pos_ahead2, self.left_border_points, self.right_border_points)
            detected_points.append(detected_point)
            distances.append(detected_point.dist)

        distances.append(normalized_speed)

        return RadarState(car_pos, car_pos_ahead, detected_points, distances)

    def get_action(self, model_network, state, epsilon, device, is_training = True) -> int:
        # Espilon-Greedy: tradeoff exploration / exploitation
        if np.random.random() < epsilon and is_training:
            move = np.random.randint(0, 5)
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = model_network(state0)
            move = torch.argmax(prediction).item()

        return move

    def check_validity(self, state: RadarState) -> bool:
        for detected_point in state.detected_points:
            if detected_point.pos == Point2D(0, 0):
                return False
            
        return True


