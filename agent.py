from collections import namedtuple
import math
import pandas as pd
import json

# The State class is a named tuple that contains all the information that the agent needs to take a decision
State = namedtuple('State',
    (
        'speed',
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
        'direction_of_second_turn'
    ))

# Get the list of points on the middle line of the track
def get_list_point_middle_line(track_name: str):
    # Read the csv file that contains the points on the middle line of the track
    path_to_csv = f'maps/{track_name}/road_middle.csv'
    df_points_on_middle_line = pd.read_csv(path_to_csv)
    return list(zip(df_points_on_middle_line.x_values, df_points_on_middle_line.y_values))

def get_road_sections(track_name: str):
    # Read the json file that contains the road sections of the track
    ## Coordinates of turns, straight lines, etc.
    ## Direction of turns, straight lines, etc.
    ## list of points on the middle line of the track related to each road section
    path_to_json = f'maps/{track_name}/dict.json'
    with open(path_to_json, "r") as json_file:
        file_data  = json.load(json_file)

    return file_data["road_sections"]

# Get the angle / orientation between two points
def get_angle_from_two_points(point, previous_point):
    return math.atan2(point[0] - previous_point[0], point[1] - previous_point[1])

def get_informations_from_turns(middle_point, road_sections):
    # Find on which section the car is located
    first_road_section_found = False
    second_road_section_found = False
    first_road_section = None
    second_road_section = None

    # Loop over all the road sections to find the two next turns
    for road_section in road_sections:

        if first_road_section_found and second_road_section_found:
            second_road_section = road_section
            break

        elif first_road_section_found and not second_road_section_found:
            first_road_section = road_section
            second_road_section_found = True
        
        elif middle_point in road_section["list_middle_points"]:
            first_road_section_found = True

    # If the car is not on a road section, return an error
    if first_road_section_found:
        if first_road_section == None:
            # We reach the end of the track so we don't have the information
            dist_1st_turn = 48
            dir_1st_turn = 0
        else:
            # Get the direction and compute the distance between the current section and the next section
            dir_1st_turn = first_road_section["direction_turn"]
            dist_1st_turn = math.sqrt((middle_point[0] - first_road_section["next_turn_point"][0])**2 + (middle_point[1] - first_road_section["next_turn_point"][1])**2)
            #TODO: Compute the distance when it's a turn
    else:
        raise Exception("The car is not on a road section")
        
    if second_road_section == None:
        # We reach the end of the track so we don't have the information
        dist_2nd_turn = 48
        dir_2nd_turn = 0
    else:
        # Get the direction and compute the distance between the current section and the next section
        dir_2nd_turn = second_road_section["direction_turn"]
        dist_2nd_turn = dist_1st_turn + math.sqrt((first_road_section["next_turn_point"][0] - second_road_section["next_turn_point"][0])**2 + (first_road_section["next_turn_point"][1] - second_road_section["next_turn_point"][1])**2)

    return dist_1st_turn, dist_2nd_turn, dir_1st_turn, dir_2nd_turn

def get_positional_informations(car_location, yaw, list_points_on_middle_line, road_sections):

    # Initialize the minimum distance to a high value and the previous point to the first point
    min_distance = 1000
    previous_point = list_points_on_middle_line[0]

    # Loop over all the points on the middle line to find the CLOSEST POINT OF THE CENTERLINE TO THE CAR
    for point in list_points_on_middle_line[1:]:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if distance > min_distance:
            break

        if distance < min_distance:
            min_distance = distance

        previous_point = point

    # Get the angle between the road's direction and car's direction
    angle = get_angle_from_two_points(point, previous_point) - yaw

    # Get distance and direction to the next turns
    dist_1st_turn, dist_2nd_turn, dir_1st_turn, dir_2nd_turn = get_informations_from_turns(point, road_sections)

    return [min_distance, angle, dist_1st_turn, dist_2nd_turn, dir_1st_turn, dir_2nd_turn]

class Agent:

    def __init__(self, track_name: str):
        self.track_name = track_name
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)
    
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


    
