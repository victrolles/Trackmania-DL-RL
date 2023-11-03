from collections import namedtuple
import math
import pandas as pd

State = namedtuple('State',
        ('speed',
        'acceleration',
        'turning_rate',
        'is_free_wheeling',
        'is_sliding',
        'as_any_lateral_contact',
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

def get_angle_from_two_points(point, previous_point):
    return math.atan2(point[0] - previous_point[0], point[1] - previous_point[1])

def get_distance_and_angle_to_centerline(car_location, yaw, list_points_on_middle_line):
    min_distance = 1000
    previous_point = list_points_on_middle_line[0]

    for point in list_points_on_middle_line[1:]:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is more than 1.5 times the minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if distance > 1.5 * min_distance:
            break

        if distance < min_distance:
            min_distance = distance

        previous_point = point

    angle = get_angle_from_two_points(point, previous_point) - yaw

    return min_distance, angle

class Agent:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.list_points_on_middle_line_of_track = get_middle_line_of_track(track_name)

    
    def get_state(self, iface_state, previous_state: State, previous_time: int, current_time: int) -> State:

        # Get the speed of the car
        speed = iface_state.display_speed

        # Get the acceleration of the car
        if previous_state is None:
            acceleration = 0
        else:
            acceleration = (speed - previous_state.speed) / (current_time - previous_time)

        # Get the turning rate of the car
        turning_rate = iface_state.scene_mobil.turning_rate

        # Get the is_free_wheeling of the car
        is_free_wheeling = iface_state.scene_mobil.is_freewheeling

        # Get the is_sliding of the car
        is_sliding = iface_state.scene_mobil.is_sliding

        # Get the has_any_lateral_contact of the car
        has_any_lateral_contact = iface_state.scene_mobil.has_any_lateral_contact

        # Get the distance and angle to the centerline
        distance_to_centerline, angle_to_centerline = get_distance_and_angle_to_centerline(iface_state.position,
                                                                                           iface_state.yaw_pitch_roll[0],
                                                                                           self.list_points_on_middle_line_of_track)


    
