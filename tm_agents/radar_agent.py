import math

import numpy as np
import torch

from config.data_classes import RadarState, Point2D, DetectedPoint

class RadarAgent:

    def __init__(self, list_points_left_border, list_points_right_border) -> None:

        self.list_points_left_border = list_points_left_border
        self.list_points_right_border = list_points_right_border

    def get_state(self, iface_state) -> RadarState:
        
        car_pos = Point2D(iface_state.position[0], iface_state.position[2])
        car_pos_ahead = point_ahead(car_pos, iface_state.yaw_pitch_roll[0])

        detected_point = closest_intersection(car_pos, car_pos_ahead, self.list_points_left_border, self.list_points_right_border)
        detected_points = [detected_point]
        distances = [detected_point.dist]

        for i in range(1, 5):
            car_pos_ahead1 = point_ahead(car_pos, iface_state.yaw_pitch_roll[0], np.pi/9*i)
            detected_point = closest_intersection(car_pos, car_pos_ahead1, self.list_points_left_border, self.list_points_right_border)
            detected_points.append(detected_point)
            distances.append(detected_point.dist)

            car_pos_ahead2 = point_ahead(car_pos, iface_state.yaw_pitch_roll[0], -np.pi/9*i)
            detected_point = closest_intersection(car_pos, car_pos_ahead2, self.list_points_left_border, self.list_points_right_border)
            detected_points.append(detected_point)
            distances.append(detected_point.dist)

        return RadarState(car_pos, car_pos_ahead, detected_points, distances)

    def get_action(self, model_network, state, epsilon, device, is_training = True) -> int:
        # Espilon-Greedy: tradeoff exploration / exploitation
        if np.random.random() < epsilon and is_training:
            move = np.random.randint(0, 5)
            # print("Random move: ", move, flush=True)
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=device)
            prediction = model_network(state0)
            move = torch.argmax(prediction).item()
            # print("Predicted move: ", move, flush=True)

        return move

# Return the intersection point between a line and a segment
def intersection_line_segment(line_p1: Point2D, line_p2: Point2D, seg_p1: Point2D, seg_p2: Point2D) -> Point2D:

    denom = (seg_p2.y - seg_p1.y) * (line_p2.x - line_p1.x) - (seg_p2.x - seg_p1.x) * (line_p2.y - line_p1.y)

    if denom == 0:
        # print("Denom = 0", flush=True)
        return None  # Les droites sont parallèles

    ua = ((seg_p2.x - seg_p1.x) * (line_p1.y - seg_p1.y) - (seg_p2.y - seg_p1.y) * (line_p1.x - seg_p1.x)) / denom
    ub = ((line_p2.x - line_p1.x) * (line_p1.y - seg_p1.y) - (line_p2.y - line_p1.y) * (line_p1.x - seg_p1.x)) / denom

    if ua >= 0 and 0 <= ub <= 1:
        # Calcul du point d'intersection
        x = line_p1.x + ua * (line_p2.x - line_p1.x)
        y = line_p1.y + ua * (line_p2.y - line_p1.y)
        return Point2D(x, y)
    if ua < 0:
        # Calcul du point d'intersection
        return Point2D(0, 0)

    if ub < 0 or ub > 1:
        # return Point2D(0, 0)
        # print("Intersection not in segment", flush=True)
        return None
    else:
        # print("No intersection", flush=True)
        # return Point2D(0, 0)
        return None

# Return the closest intersection point between a line and the borders of the track 
def find_closest_intersection(line_p1: Point2D, line_p2: Point2D, list_points_border) -> Point2D:

    min_dist = float('inf')
    closest_point = None

    for i in range(len(list_points_border) - 1):
        border_p1 = list_points_border[i]
        border_p1_ = Point2D(border_p1[0], border_p1[1])
        border_p2 = list_points_border[i + 1]
        border_p2_ = Point2D(border_p2[0], border_p2[1])
        point = intersection_line_segment(line_p1, line_p2, border_p1_, border_p2_)
        if point:
            dist = np.linalg.norm(point.to_array() - line_p1.to_array())
            if dist < min_dist:
                min_dist = dist
                closest_point = point
    
    if closest_point == None:
        print("No intersection found", flush=True)
    return closest_point

def closest_intersection(car_pos: Point2D, car_pos_ahead: Point2D, list_points_left_border, list_points_right_border) -> DetectedPoint:
    """
    Retourne le point d'intersection le plus proche entre la droite passant par la voiture
    et les bords de la route.
    """
    line_p1 = car_pos
    line_p2 = car_pos_ahead

    # Trouver l'intersection la plus proche pour les deux bordures
    closest_left = find_closest_intersection(line_p1, line_p2, list_points_left_border)
    closest_right = find_closest_intersection(line_p1, line_p2, list_points_right_border)

    # Comparer les deux intersections trouvées et retourner la plus proche
    if closest_left and closest_right:
        dist_left = closest_left.distance(line_p1)
        dist_right = closest_right.distance(line_p1)
        if dist_left < dist_right:
            return DetectedPoint(closest_left, dist_left)
        else:
            return DetectedPoint(closest_right, dist_right)
    elif closest_left:
        dist_left = closest_left.distance(line_p1)
        return DetectedPoint(closest_left, dist_left)
    elif closest_right:
        dist_right = closest_right.distance(line_p1)
        return DetectedPoint(closest_right, dist_right)
    else:
        return None

def point_ahead(car_pos: Point2D, theta, extra_rotation = 0) -> Point2D:
    """
    Retourne un point 1 unité devant la voiture donnée sa position (x, y) et son angle de rotation theta.
    
    :param x: Position x de la voiture
    :param y: Position y de la voiture
    :param theta: Angle de rotation de la voiture en radians (entre -pi et pi)
    :return: Tuple (x_new, y_new) représentant le point 1 unité devant la voiture
    """
    # Ajuster l'angle pour pointer devant la voiture
    theta_adjusted = - theta + np.pi / 2 + extra_rotation

    # Calcul du déplacement en x et y
    dx = math.cos(theta_adjusted)
    dy = math.sin(theta_adjusted)
    
    # Calcul de la nouvelle position
    x_new = car_pos.x + 2*dx
    y_new = car_pos.y + 2*dy
    
    return Point2D(x_new, y_new)