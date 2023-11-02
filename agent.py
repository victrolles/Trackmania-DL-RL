from collections import namedtuple
import math
import pandas as pd

State = namedtuple('State',
        ('speed',
        'acceleration',
        'distance_to_centerline',
        'angle_to_centerline',
        'distance_to_first_turn',
        'distance_to_second_turn',
        'direction_of_first_turn',
        'direction_of_second_turn'))

def get_middle_line_of_track(track_name: str):
    path_to_csv = f'maps/{track_name}/road_middle.csv'
    df_points_on_middle_line = pd.read_csv(path_to_csv)
    return list(zip(df_points_on_middle_line.x_values, df_points_on_middle_line.y_values))

def get_closest_point_to_middle_line(car_location, list_points_on_middle_line):
    min_distance = 1000

    for point in list_points_on_middle_line:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is more than 1.5 times the minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if distance > 1.5*min_distance:
            return min_distance

        if distance < min_distance:
            min_distance = distance

    return min_distance

class Agent:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.list_points_on_middle_line_of_track = get_middle_line_of_track(track_name)

    
    def get_state(self, iface_state, previous_state: State, previous_time: int, current_time: int) -> State:

        # Get the speed of the car
        speed = math.sqrt(iface_state.velocity[0]**2 + iface_state.velocity[1]**2 + iface_state.velocity[2]**2)

        # Get the acceleration of the car
        if previous_state is None:
            acceleration = 0
        else:
            acceleration = (speed - previous_state.speed) / (current_time - previous_time)

        # Get the distance to the centerline
        distance_to_centerline = get_closest_point_to_middle_line(
            iface_state.position,
            self.list_points_on_middle_line_of_track
            )
        
        # Get the angle to the centerline
        

    
