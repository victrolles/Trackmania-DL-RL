from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import csv


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.is_first_iter = True
        self.x_values = []
        self.y_values = []

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            zipped = zip(self.x_values, self.y_values)
            with open('maps/snake_map_training/road_left.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(zipped)
            iface.close()

    def on_run_step(self, iface: TMInterface, _time: int):
        if self.is_first_iter:
            self.is_first_iter = False
            print("First iteration")
            iface.set_speed(1)

        if _time >= 0:
            state = iface.get_simulation_state()
            self.x_values.append(state.position[0])
            self.y_values.append(state.position[2])

            print(f'iter : {len(self.x_values)}, time : {_time}', end='\r')

   

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()