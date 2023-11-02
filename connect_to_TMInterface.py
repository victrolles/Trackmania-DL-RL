from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import time
import math
from collections import namedtuple

State = namedtuple('State',
    ('speed',
        'acceleration',
        'distance_to_centerline',
        'angle_to_centerline',
        'distance_to_first_turn',
        'distance_to_second_turn',
        'direction_of_first_turn',
        'direction_of_second_turn'
    )
)

def get_state(iface_state, previous_state: State, previous_time: int, current_time: int) -> State:

    speed = math.sqrt(iface_state.velocity[0]**2 + iface_state.velocity[1]**2 + iface_state.velocity[2]**2)

    if previous_state is None:
        acceleration = 0
    else:
        acceleration = (speed - previous_state.speed) / (current_time - previous_time)

    
def get_closest_point_to_middle_line(x_y_car_location, list_points_on_middle_line):
    min_distance = 1000
    closest_point = None

    for point in list_points_on_middle_line:
        distance = math.sqrt((x_y_car_location[0] - point[0])**2 + (x_y_car_location[1] - point[1])**2)

        # If the distance is more than 1.5 times the minimum distance, we can stop searching
        # because we reach the closest point and we are now going away from it
        if distance > 1.5*min_distance:
            return min_distance

        if distance < min_distance:
            min_distance = distance
            closest_point = point

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.iter = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.close()

    def on_run_step(self, iface: TMInterface, _time: int):

        

        self.iter += 1
        # iface.set_speed(2)
        if _time >= 0:


            inputs = {  # 0 Forward
                "left": False,
                "right": False,
                "accelerate": True,
                "brake": False,
            }
            iface.set_input_state(**inputs)
            state = iface.get_simulation_state()
            print(f'iter : {self.iter}, time : {_time}', end='\r')

            # if _time > 20000:
            #     iface.give_up()

            # if _time > 30000:
            #     iface.close()

            # print(
            #     f'Time: {_time}\n'
            #     f'Display Speed: {state.display_speed}\n'
            #     f'Position: {state.position}\n'
            #     f'Velocity: {state.velocity}\n'
            #     f'YPW: {state.yaw_pitch_roll}\n'
            # , end='\r')
        # print(f'on_run_step, time : {time.time()}', end="\r")

   

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()