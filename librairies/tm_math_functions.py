import math
import json
import numpy as np
import pandas as pd

from librairies.data_classes import Point2D, DetectedPoint
from librairies.dictionaries import Rd

def delta_time_to_str(delta_time: float) -> str:

    hour = int(delta_time / 3600)
    minute = int(delta_time / 60)
    seconde = int(delta_time % 60)

    if hour > 0:
        string = f"{hour}h {minute}min {seconde}s"
    elif minute > 0:
        string = f"{minute}min {seconde}s"
    else:
        string = f"{seconde}s"

    return string

def get_closest_point(car_pos: Point2D, middle_points: list[Point2D]) -> Point2D:
    
    min_distance = float('inf')
    closest_point = None

    for point in middle_points:

        distance = car_pos.distance(point)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if (distance > 1.1 * min_distance) and (min_distance < 12.0): #1.05
            break

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

def get_next_point(closest_point: Point2D, middle_points: list[Point2D]) -> Point2D:

    try:
        idx = middle_points.index(closest_point)
    except ValueError:
        raise ValueError("The closest_point is not in the middle_points list.")

    if idx == 0:
        return middle_points[1]
    else:
        return middle_points[idx - 1]

def distance_to_finish_line(car_pos: Point2D, closest_point: Point2D, middle_points: list[Point2D]):

    try:
        idx = middle_points.index(closest_point)
    except ValueError:
        raise ValueError("The closest_point is not in the middle_points list.")
    
    dist_to_finish_line = car_pos.distance(closest_point) 

    for i in range(idx, len(middle_points) - 1):
        dist_to_finish_line += middle_points[i].distance(middle_points[i + 1])

    return dist_to_finish_line

def get_road_sections(track_name: str):
    path_to_json = f'extras/maps/{track_name}/dict.json'
    with open(path_to_json, "r") as json_file:
        file_data  = json.load(json_file)

    return file_data["road_sections"]

def get_road_df(track_name: str, road_side: Rd, shorter_list: bool = False) -> pd.DataFrame:
    path_to_csv = f'extras/maps/{track_name}/road_{road_side.value}.csv'
    df = pd.read_csv(path_to_csv)

    if shorter_list:
        df = df.iloc[::20]

    return df

def get_road_points(track_name: str, road_side: Rd, shorter_list: bool = False, print_size: bool = False) -> list[Point2D]:
    path_to_csv = f'extras/maps/{track_name}/road_{road_side.value}.csv'

    # Read the CSV file
    df = pd.read_csv(path_to_csv)
    
    # Ensure the columns are named 'x_values' and 'y_values'
    if 'x_values' not in df.columns or 'y_values' not in df.columns:
        raise ValueError("CSV must contain 'x_values' and 'y_values' columns")
    
    # Convert the columns to a list of Point2D
    list_points = [Point2D(x, y) for x, y in zip(df['x_values'], df['y_values'])]

    if shorter_list:
        list_points = list_points[::20]

    if print_size:
        print(f"Size list {road_side} : {len(list_points)}", flush=True)
    
    return list_points

def distance_point_to_line(point: Point2D, line_p1: Point2D, line_p2: Point2D) -> float:
    x0, y0 = point.x, point.y
    x1, y1 = line_p1.x, line_p1.y
    x2, y2 = line_p2.x, line_p2.y
    
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return numerator / denominator

def angle_between_lines(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D) -> float:
    # Direction vectors of the lines
    v1 = p2.to_array() - p1.to_array()
    v2 = q2.to_array() - q1.to_array()
    
    # Dot product and norms of the vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Angle in radians
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Ensure the value is within the valid range for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta = np.arccos(cos_theta)
    
    return theta

# Return the intersection point between a line and a segment
def intersection_line_segment(line_p1: Point2D, line_p2: Point2D, seg_p1: Point2D, seg_p2: Point2D) -> Point2D:

    denom = (seg_p2.y - seg_p1.y) * (line_p2.x - line_p1.x) - (seg_p2.x - seg_p1.x) * (line_p2.y - line_p1.y)

    if denom == 0:
        return None  # lines are parallel

    ua = ((seg_p2.x - seg_p1.x) * (line_p1.y - seg_p1.y) - (seg_p2.y - seg_p1.y) * (line_p1.x - seg_p1.x)) / denom
    ub = ((line_p2.x - line_p1.x) * (line_p1.y - seg_p1.y) - (line_p2.y - line_p1.y) * (line_p1.x - seg_p1.x)) / denom

    if ua >= 0 and 0 <= ub <= 1:
        # Calcul du point d'intersection
        x = line_p1.x + ua * (line_p2.x - line_p1.x)
        y = line_p1.y + ua * (line_p2.y - line_p1.y)
        return Point2D(x, y)
    
    if ua < 0:
        return Point2D(0, 0) # Intersection not in semi-infinite line

    if ub < 0 or ub > 1:
        return None # Intersection not in segment
    
    else:
        return None # No intersection

# Return the closest intersection point between a line and the borders of the track 
def find_closest_intersection(line_p1: Point2D, line_p2: Point2D, border_points: list[Point2D]) -> Point2D:

    min_dist = float('inf')
    closest_point = None

    for i in range(len(border_points) - 1):

        border_p1 = border_points[i]
        border_p2 = border_points[i + 1]

        point = intersection_line_segment(line_p1, line_p2, border_p1, border_p2)

        if point:

            dist = point.distance(line_p1)

            if dist < min_dist:
                min_dist = dist
                closest_point = point
    
    if closest_point == None:
        print("No intersection found", flush=True)

    return closest_point

def closest_intersection(car_pos: Point2D, car_pos_ahead: Point2D, left_border_points: list[Point2D], right_border_points: list[Point2D]) -> DetectedPoint:
    """
    Retourne le point d'intersection le plus proche entre la droite passant par la voiture
    et les bords de la route.
    """
    line_p1 = car_pos
    line_p2 = car_pos_ahead

    # Trouver l'intersection la plus proche pour les deux bordures
    closest_left = find_closest_intersection(line_p1, line_p2, left_border_points)
    closest_right = find_closest_intersection(line_p1, line_p2, right_border_points)

    # Comparer les deux intersections trouvées et retourner la plus proche
    if closest_left and closest_right:

        dist_left = closest_left.distance(line_p1, True)
        dist_right = closest_right.distance(line_p1, True)

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

def point_ahead(car_pos: Point2D, theta: float, extra_rotation: float = 0) -> Point2D:
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