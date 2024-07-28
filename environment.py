import sys

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import pandas as pd
import numpy as np

from utils import point_ahead, closest_intersection
from config.data_classes import Point2D, DataBus

class Environment(Client):
    def __init__(self, databus_buffer: DataBus) -> None:
        super(Environment, self).__init__()

        ## Shared memory
        self.databus_buffer = databus_buffer

        # Training state
        # self.x = x
        # self.y = y
        # self.x2 = x2
        # self.y2 = y2
        # self.x3 = x3
        # self.y3 = y3

        # self.previous_x = 0
        # self.previous_y = 0

        self.iter = 0

        folder = 'snake_map_training'
        list_points_left_border = pd.read_csv(f'maps/{folder}/road_left.csv')
        list_points_right_border = pd.read_csv(f'maps/{folder}/road_right.csv')
        list_points_left_border = list_points_left_border.iloc[::25]
        list_points_right_border = list_points_right_border.iloc[::25]
        self.list_points_left_border = list(zip(list_points_left_border.x_values, list_points_left_border.y_values))
        self.list_points_right_border = list(zip(list_points_right_border.x_values, list_points_right_border.y_values))
        print("size left: ", len(self.list_points_left_border))
        print("size right: ", len(self.list_points_right_border))

        # ax.set(xlabel='x', ylabel='y')
        # ax.legend()
        # plt.show()

        # # Setup screen and game
        # def set_window_pos(window_name, x, y, width, height):
        #     hwnd = win32gui.FindWindow(None, window_name)
        #     if hwnd:
        #         # print(win32gui.GetWindowRect(hwnd))
        #         win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, width, height, 0)

        # # Example usage
        # name = "TrackMania Nations Forever (TMInterface 1.4.3)"
        # set_window_pos(name, -6, 0, 256, 256)  # Replace 'Google Chrome' with the exact window title

        # ## To sort out
        # track_name = str('snake_map_training')
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.game_experience = []

        # self.policy_model = PolicyModel(12, 5).to(self.device) #400, 512, 3
        # self.q1_model = QModel(12, 5).to(self.device)
        # self.q2_model = QModel(12, 5).to(self.device)

        # self.agent = Agent(track_name)
        # self.dqn_trainer = SACTrainer(self.policy_model, self.q1_model, self.q2_model, self.device, self.is_model_saved, self.end_processes, track_name, self.episode, self.policy_loss, self.q1_loss, self.q2_loss, self.training_time)
        # self.dqn_trainer = DQNTrainer(self.game_experience, self.epsilon, self.epoch, self.loss, self.device, self.is_model_saved, self.end_processes, track_name, self.training_time)
        # self.inactivity = 0
        # self.is_track_finished = bool(False)
        # self.current_game_speed = 1.0
        
        # self.track_name = track_name
        # self.list_point_middle_line = get_list_point_middle_line(track_name)
        # self.road_sections = get_road_sections(track_name)

        # self.previous_dist_to_finish_line = 1103.422# 917.422
        # self.previous_state = None
        # self.previous_action = None

    # Connection to Trackmania
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}', flush=True)
        iface.set_timeout(20_000)
        iface.give_up()

    # Function to detect if the car crossed the finish line
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()
            self.is_track_finished = True
            iface.give_up()

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time >= 0:
            self.iter += 1
            # Get the state of the car
            
            # print(f"Position: {state.position[0]}, {state.position[1]}")

            if self.iter % 30 == 0:
                state = iface.get_simulation_state()

                car_pos = Point2D(state.position[0], state.position[2])
                car_pos_ahead = point_ahead(car_pos, state.yaw_pitch_roll[0])

                intersect_point = closest_intersection(car_pos, car_pos_ahead, self.list_points_left_border, self.list_points_right_border)
                intersect_points = [intersect_point]

                for i in range(1, 3):
                    car_pos_ahead1 = point_ahead(car_pos, state.yaw_pitch_roll[0], np.pi/6*i)
                    intersect_point = closest_intersection(car_pos, car_pos_ahead1, self.list_points_left_border, self.list_points_right_border)
                    intersect_points.append(intersect_point)

                    car_pos_ahead2 = point_ahead(car_pos, state.yaw_pitch_roll[0], -np.pi/6*i)
                    intersect_point = closest_intersection(car_pos, car_pos_ahead2, self.list_points_left_border, self.list_points_right_border)
                    intersect_points.append(intersect_point)

                data_bus = DataBus(car_pos, car_pos_ahead, intersect_points)
                self.databus_buffer.put(data_bus)

            # # ===== Stop the process if needed =====
            # if self.end_processes.value:
            #     # Close the connection to Trackmania
            #     iface.close()
            #     # Save the model
            #     # self.dqn_trainer.save_model()
            #     print("Environment process correctly stopped")
            #     return
            

            
            
                
def start_env(databus_buffer):
    print("Environment process started")
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Environment(databus_buffer))