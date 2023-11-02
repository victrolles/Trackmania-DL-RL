from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import time
import math
from collections import namedtuple



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

        if _time >= 0:

            self.iter += 1
            state = iface.get_simulation_state()

            # print(
            #     f'Time: {_time}\n'
            #     f'Display Speed: {state.display_speed}\n'
            #     f'Position: {state.position}\n'
            #     f'Velocity: {state.velocity}\n'
            #     f'YPW: {state.yaw_pitch_roll}\n'
            #     # f'state player info0: {state.player_info[0]}\n'
            #     # f'state player info1: {state.player_info[1]}\n's
            #     f'state player race finished: {state.player_info.race_finished}\n'
            #     # f'state wheels: {state.simulation_wheels}\n'
            #     f'state wheels 0 steerable: {state.simulation_wheels[0].steerable}\n'
            #     f'state wheels 0 has_ground_contact: {state.simulation_wheels[0].real_time_state.has_ground_contact}\n'
            #     f'state wheels 0 is_sliding: {state.simulation_wheels[0].real_time_state.is_sliding}\n'
            #     f'state scene_mobil has_any_lateral_contact: {state.scene_mobil.has_any_lateral_contact}\n'
            #     f'state scene_mobil turning_rate: {state.scene_mobil.turning_rate}\n'
            #     f'state scene_mobil is_freewheeling: {state.scene_mobil.is_freewheeling}\n'
            #     f'state scene_mobil is_sliding : {state.scene_mobil}\n'
            # , end='\r')

            print(f'Speed: {state.display_speed}, turning rate : {state.scene_mobil.turning_rate}', end='\r')

            if self.iter > 10000:
                iface.close()
   

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()