import time

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from config.data_classes import DataBus, Point2D, TrainingStats, RadarState
from config.globals import TRACK_NAME

class Graphic:
    def __init__(self,
                 databus_buffer: DataBus,
                 end_processes,
                 tm_speed,
                 is_training,
                 save_model,
                 is_map_render,
                 is_curves_render,
                 is_tm_speed_changed) -> None:

        print("Graphic process started", flush=True)

        ## Bus data
        self.databus_buffer = databus_buffer

        ## Shared memory
        self.end_processes = end_processes
        self.tm_speed = tm_speed
        self.is_training = is_training
        self.save_model = save_model
        self.is_map_render = is_map_render
        self.is_curves_render = is_curves_render
        self.is_tm_speed_changed = is_tm_speed_changed

        self.iter = 0
        self.start_time = time.time()

        # Constants
        self.root = tk.Tk()
        self.root.title("Car Racing Board")
        self.root.geometry(str(1200) + "x" + str(800))

        self.init_canvas()

        ## Main loop
        self.update_display()


    # Update display
    def update_display(self):

        while not self.end_processes.value:

            time.sleep(0.05)
            self.iter += 1
            
            if not self.databus_buffer.empty():

                # ---- Get data ----
                databus: DataBus = self.databus_buffer.get()

                # ---- Update displays ----
                # Map
                if databus.radar_state is not None:
                    self.plot_map.update_infos(databus.radar_state)

                # Curves
                if databus.training_stats is not None and databus.distance_travelled != 0.0:
                    self.plot_curves.update_infos(databus.training_stats, databus.distance_travelled)

                # Stats
                if databus.training_stats is not None:
                    self.update_training_stats(databus.training_stats)

                if databus.total_time is not None:
                    self.update_total_time(databus.total_time)

                # FPS
                self.update_fps(int(databus.fps_env), int(self.iter / (time.time() - self.start_time)))

                # Buffers
                self.update_buffers(databus.exp_buffer_size, self.databus_buffer.qsize())

                # Check buttons
                self.check_speed()
                self.check_save_model()

                # Update display
                self.root.update()

        print("Graphic process correctly stopped", flush=True)

    def init_canvas(self):
        # canvas
        ## plots:
        self.plot_map = PlotMap(self.root)
        self.plot_curves = PlotCurves(self.root)

        ## stats:
        #infos
        self.label_fps_env = tk.Label(self.root, text=f"fps env: {0}",font=("Arial", 20))
        self.label_fps_graphic = tk.Label(self.root, text=f"fps graph: {0}",font=("Arial", 20))

        self.label_epoch = tk.Label(self.root, text=f"Epoch: {0}", font=("Arial", 20))
        self.label_step = tk.Label(self.root, text=f"Step: {0}", font=("Arial", 20))

        self.label_total_time = tk.Label(self.root, text=f" Total Time: {0:.3f}", font=("Arial", 20))
        self.label_training_time = tk.Label(self.root, text=f"Training Time: {0:.3f}", font=("Arial", 20))

        self.label_size_exp_buffer = tk.Label(self.root, text=f"Exp buffer size: {0}", font=("Arial", 20))
        self.label_size_bus_buffer = tk.Label(self.root, text=f"Bus buffer size: {0}", font=("Arial", 20))

        self.label_epsilon = tk.Label(self.root, text=f"Epsilon: {0:.3f}", font=("Arial", 20))

        # buttons
        self.label_button_speed = tk.Label(self.root, text='Game speed: 1.0', font=("Arial", 20))
        self.label_button_training = tk.Label(self.root, text="Mode: Training", font=("Arial", 20))
        self.label_button_save_model = tk.Label(self.root, text="Save models", font=("Arial", 20))
        self.label_button_render_map = tk.Label(self.root, text="Render map:", font=("Arial", 20))
        self.label_button_render_curves = tk.Label(self.root, text="Render curves:", font=("Arial", 20))

        ## buttons:
        self.button_training = tk.Button(self.root, text="Switch mode", command=self.switch_mode, font=("Arial", 20))
        self.button_save_model = tk.Button(self.root, text="Save models", command=self.action_save_model, font=("Arial", 20))
        self.button_speed = tk.Button(self.root, text="Set", command=self.change_game_speed, font=("Arial", 20))
        self.button_render_map = tk.Button(self.root, text="On", command=self.change_render_map, font=("Arial", 20))
        self.button_render_curves = tk.Button(self.root, text="On", command=self.change_render_curves, font=("Arial", 20))
        self.button_exit = tk.Button(self.root, text="Exit", command=self.exit, font=("Arial", 20))

        ## Entry
        self.entry_game_speed = tk.Entry(self.root, bg="#FFFFFF", fg="#000000", font=("Arial", 20), width=8)

        # Place
        self.label_fps_env.place(x=450, y=5)
        self.label_fps_graphic.place(x=450, y=45)
        self.label_epoch.place(x=450, y=85)
        self.label_step.place(x=450, y=125)

        self.label_total_time.place(x=750, y=5)
        self.label_training_time.place(x=750, y=45)
        self.label_epsilon.place(x=750, y=85)
        self.label_size_exp_buffer.place(x=750, y=125)
        self.label_size_bus_buffer.place(x=750, y=165)

        self.label_button_training.place(x=450, y=220)
        self.label_button_save_model.place(x=450, y=260)
        self.label_button_speed.place(x=450, y=300)
        self.entry_game_speed.place(x=450, y=340)

        self.button_training.place(x=650, y=210)
        self.button_save_model.place(x=650, y=250)
        self.button_speed.place(x=600, y=340)

        self.label_button_render_map.place(x=900, y=300)
        self.label_button_render_curves.place(x=900, y=340)

        self.button_render_map.place(x=1100, y=290)
        self.button_render_curves.place(x=1100, y=330)
        self.button_exit.place(x=1100, y=65)

    # one way buttons
    def exit(self):
        self.end_processes.value = True
        self.root.quit()
    
    def change_game_speed(self):
        self.is_tm_speed_changed.value = True
        self.tm_speed.value = float(self.entry_game_speed.get())
        self.entry_game_speed.delete(0, 'end')
        self.label_button_speed.config(text="Game speed : switching ...")

    def action_save_model(self):
        self.save_model.value = True
        self.label_button_save_model.config(text="Saving models ...")

    def check_save_model(self):
        if not self.save_model.value and self.label_button_save_model.cget("text") == "Saving models ...":
            self.label_button_save_model.config(text="Save models")

    def check_speed(self):
        if not self.is_tm_speed_changed.value and self.label_button_speed.cget("text") == "Game speed : switching ...":
            self.label_button_speed.config(text=f"Game speed: {self.tm_speed.value}")

    # switch mode buttons
    def change_render_map(self):
        self.is_map_render.value = not self.is_map_render.value
        self.button_render_map.config(text="On" if self.is_map_render.value else "Off")

    def change_render_curves(self):
        self.is_curves_render.value = not self.is_curves_render.value
        self.button_render_curves.config(text="On" if self.is_curves_render.value else "Off")

    def switch_mode(self):
        self.is_training.value = not self.is_training.value
        self.label_button_training.config(text="Mode: Training" if self.is_training.value else "Mode: Testing")

    

    def update_total_time(self, total_time: float):
        self.label_total_time.config(text=f"Total Time: {delta_time_to_str(total_time)}")

    def update_training_stats(self, training_stats: TrainingStats):
        self.label_training_time.config(text=f"Training Time: {delta_time_to_str(training_stats.training_time)}")
        self.label_epoch.config(text=f"Epoch: {training_stats.epoch}")
        self.label_step.config(text=f"Step: {training_stats.step}")
        self.label_epsilon.config(text=f"Epsilon: {training_stats.epsilon:.3f}")

    def update_fps(self, fps_env: float, fps_graphic: float):
        self.label_fps_env.config(text=f"fps env: {fps_env}")
        self.label_fps_graphic.config(text=f"fps graph: {fps_graphic}")

    def update_buffers(self, exp_buffer_size: int, bus_buffer_size: int):
        self.label_size_exp_buffer.config(text=f"Exp buffer size: {exp_buffer_size}")
        self.label_size_bus_buffer.config(text=f"Bus buffer size: {bus_buffer_size}")

def delta_time_to_str(delta_time: float):
    minute = int(delta_time / 60)
    seconde = int(delta_time % 60)
    string = f"{minute}min {seconde}s"
    return string


class PlotCurves:
    def __init__(self, root):
        self.root = root

        self.fig, (self.graph_distances, self.graph_losses) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.suptitle('Training Curves')

        self.list_epoch = []
        self.list_distance_traveled = []
        self.list_loss = []

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(x=420, y=400)

    def update_infos(self, training_stats: TrainingStats, distance_traveled: float):
        self.list_epoch.append(training_stats.epoch)
        self.list_distance_traveled.append(distance_traveled)
        self.list_loss.append(training_stats.loss)

        self.update_plot()

    def update_plot(self):
        # Update graph distances
        self.graph_distances.clear()
        self.graph_distances.set_title('Distances :')
        self.graph_distances.set_xlabel('Tries')
        self.graph_distances.set_ylabel('Distance')
        self.graph_distances.plot(self.list_epoch, self.list_distance_traveled)
        self.graph_distances.set_ylim(ymin=0)
        self.graph_distances.text(len(self.list_distance_traveled)-1, self.list_distance_traveled[-1], str(int(self.list_distance_traveled[-1])))

        # Update graph losses
        self.graph_losses.clear()
        self.graph_losses.set_title('Losses :')
        self.graph_losses.set_xlabel('Epochs')
        self.graph_losses.set_ylabel('Loss')
        self.graph_losses.plot(self.list_epoch, self.list_loss)
        self.graph_losses.set_ylim(ymin=0)
        self.graph_losses.text(len(self.list_loss)-1, self.list_loss[-1], "{:.2f}".format(self.list_loss[-1]))

        # Draw
        self.canvas.draw()
        self.canvas.flush_events()

class PlotMap:
    def __init__(self, root):
        self.root = root

        self.left_side_road_coordinates = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_left.csv')
        self.right_side_road_coordinates = pd.read_csv(f'extras/maps/{TRACK_NAME}/road_right.csv')

        self.fig, self.map = plt.subplots(figsize=(4, 6))
        self.fig.suptitle('Training Curves')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(x=30, y=30)

    def update_infos(self, radar_state: RadarState):

        self.update_plot(radar_state)

    def update_plot(self, radar_state: RadarState):
        # ---- Plot interface ----

        # Clear
        self.map.clear()

        # Road
        self.map.plot(self.left_side_road_coordinates.x_values, self.left_side_road_coordinates.y_values, color='black', label='left_side')
        self.map.plot(self.right_side_road_coordinates.x_values, self.right_side_road_coordinates.y_values, color='red', label='right_side')

        # Car
        self.map.plot(radar_state.car_pos.x, radar_state.car_pos.y, 'bo', label='car')
        self.map.plot(radar_state.car_ahead_pos.x, radar_state.car_ahead_pos.y, 'ro', label='car2')

        # Sensors
        for detected_point in radar_state.detected_points:
            print(detected_point.dist, flush=True)
            color = plt.cm.RdYlGn(detected_point.dist)
            print(color, flush=True)
            self.map.plot(detected_point.pos.x, detected_point.pos.y, 'o', color=color, label='sensor')

        # ---- Draw ----
        self.canvas.draw()
        self.canvas.flush_events()