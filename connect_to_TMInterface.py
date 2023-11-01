from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import time


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