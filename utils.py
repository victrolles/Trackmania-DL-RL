import math
import pandas as pd
import json
import numpy as np

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
    for idx, road_section in enumerate(road_sections):

        if first_road_section_found and second_road_section_found:
            second_road_section = road_section
            break

        elif first_road_section_found and not second_road_section_found:
            first_road_section = road_section
            second_road_section_found = True
        
        elif list(middle_point) in road_section["list_middle_points"]:
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
    min_point = None
    min_previous_point = None

    # Loop over all the points on the middle line to find the CLOSEST POINT OF THE CENTERLINE TO THE CAR
    for point in list_points_on_middle_line[1:]:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if (distance > 1.05 * min_distance) and (min_distance < 12.0):
            break

        if distance < min_distance:
            min_distance = distance
            min_point = point
            min_previous_point = previous_point

        previous_point = point

    if min_point and min_previous_point:
        # Vector from the closest point to the next point on the middle line
        vector_a = (min_point[0] - min_previous_point[0], min_point[1] - min_previous_point[1])
        # Vector from the closest point on the middle line to the car
        vector_b = (car_location[0] - min_previous_point[0], car_location[2] - min_previous_point[1])
        
        # Cross product to determine the side
        cross_product_z = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        
        # Assign positive or negative based on the side
        final_distance = min_distance * math.copysign(1, cross_product_z)

    # Get the angle between the road's direction and car's direction
    angle = calculate_relative_angle(yaw, get_angle_from_two_points(min_point, min_previous_point))

    # Get distance and direction to the next turns
    dist_1st_turn, dist_2nd_turn, dir_1st_turn, dir_2nd_turn = get_informations_from_turns(min_point, road_sections)

    return [final_distance, angle, dist_1st_turn, dist_2nd_turn, dir_1st_turn, dir_2nd_turn]

def get_distance_to_finish_line(car_location, list_points_on_middle_line, road_sections):

    # Initialize the minimum distance to a high value
    min_distance = 1000
    min_point = None
    
    dist_to_finish_line = 0

    road_section_found = False
    in_section_dist_calculated = False

    # Loop over all the points on the middle line to find the CLOSEST POINT OF THE CENTERLINE TO THE CAR
    for point in list_points_on_middle_line[1:]:
        distance = math.sqrt((car_location[0] - point[0])**2 + (car_location[2] - point[1])**2)

        # If the distance is longer than minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if (distance > 1.05 * min_distance) and (min_distance < 12.0):
            break

        if distance < min_distance:
            min_distance = distance
            min_point = point

    # Get the road section where the car is located
    for idx, road_section in enumerate(road_sections):

        if road_section_found and in_section_dist_calculated:
            x_section = road_sections[idx]["next_turn_point"][0]
            y_section = road_sections[idx]["next_turn_point"][1]
            x_previous_section = road_sections[idx-1]["next_turn_point"][0]
            y_previous_section = road_sections[idx-1]["next_turn_point"][1]
            dist_to_finish_line += math.sqrt((x_previous_section - x_section)**2 + (y_previous_section - y_section)**2)

        elif road_section_found:
            x_section = road_section["next_turn_point"][0]
            y_section = road_section["next_turn_point"][1]
            dist_to_finish_line = math.sqrt((min_point[0] - x_section)**2 + (min_point[1] - y_section)**2)
            in_section_dist_calculated = True

        elif list(min_point) in road_section["list_middle_points"]:
            road_section_found = True

    # Check if the section is found and if the distance is calculated
            
    # Get the distance to the finish line
    if road_section_found and not in_section_dist_calculated:
        print("road_section_error")
        x_section = road_sections[-1]["next_turn_point"][0]
        y_section = road_sections[-1]["next_turn_point"][1]
        dist_to_finish_line += math.sqrt((min_point[0] - x_section)**2 + (min_point[1] - y_section)**2)

    # If the car is not on a road section, return an error
    elif not road_section_found:
        raise Exception("The road section was not found")

    return dist_to_finish_line

def convert_seconds(seconds):
    hours = seconds // 3600  # Find the whole hours
    minutes = (seconds % 3600) // 60  # Find the remaining minutes
    seconds = seconds % 60  # Find the remaining seconds
    return hours, minutes, seconds

def calculate_relative_angle(car_angle, track_angle):
    # Convert angles to vectors
    car_vector = (np.cos(car_angle), np.sin(car_angle))
    track_vector = (np.cos(track_angle), np.sin(track_angle))
    
    # Calculate the dot product and the determinant (for the 'signed' area)
    dot_product = car_vector[0] * track_vector[0] + car_vector[1] * track_vector[1]
    determinant = car_vector[0] * track_vector[1] - car_vector[1] * track_vector[0]
    
    # Calculate the angle using arctan2
    angle = np.arctan2(determinant, dot_product)
    
    return angle