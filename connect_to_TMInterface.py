from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import time
import math
from collections import namedtuple
from agent import Agent, State



class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
    
        self.agent = Agent('RL_map_training')

        self.previous_time = int(0)
        self.previous_state = None
        self.iter = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.close()

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time >= 0:
            
            # Loop : get the current state of the car
            iface_state = iface.get_simulation_state()
            current_state = self.agent.get_state(iface_state, self.previous_state, self.previous_time, _time)
            
            # Print the current state of the car
            # print(f"Spd: {current_state.speed:.1f}, Acc: {current_state.acceleration:.1f}, Turn: {current_state.turning_rate:.1f}, Free: {current_state.is_free_wheeling}, Slide: {current_state.is_sliding}, Lat: {current_state.has_any_lateral_contact}",end='\r')
            # print(f"Dist: {current_state.distance_to_centerline:.3f}, Angle: {current_state.angle_to_centerline:.3f}",end='\r')
            # print(f"Dist1: {current_state.distance_to_first_turn:.3f}, Dist2: {current_state.distance_to_second_turn:.3f}, Dir1: {current_state.direction_of_first_turn:.3f}, Dir2: {current_state.direction_of_second_turn:.3f}", end='\r')
            
            
            self.previous_time = _time
            self.previous_state = current_state
            self.iter += 1

            if self.iter > 10000:
                iface.close()
   

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()