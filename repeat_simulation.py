from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
from utils import get_distance_to_finish_line, get_list_point_middle_line, get_road_sections

class MainClient(Client):
    def __init__(self) -> None:
        track_name = 'RL_map_training'
        self.track_name = track_name
        self.list_point_middle_line = get_list_point_middle_line(track_name)
        self.road_sections = get_road_sections(track_name)
        self.state = None
        self.finished = False
        self.race_time = 0
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_simulation_begin(self, iface: TMInterface):
        iface.remove_state_validation()
        self.finished = False

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0:
            iface_state = iface.get_simulation_state()
            distance = get_distance_to_finish_line(iface_state.position,
                self.list_point_middle_line,
                self.road_sections
           )
            print("race distance: ", distance, end='\r')

    def on_simulation_step(self, iface: TMInterface, _time: int):
        self.race_time = _time
        if self.race_time == 0:
            self.state = iface.get_simulation_state()
            
        

        if self.finished:
            iface.rewind_to_state(self.state)
            self.finished = False
            

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            print(f'Finished the race at {self.race_time}')
            self.finished = True
            # iface.prevent_simulation_finish()
            # iface.give_up()

    def on_simulation_end(self, iface, result: int):
        print('Simulation finished')


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()