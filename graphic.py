import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from config.data_classes import DataBus, Point2D

class Graphic:
    def __init__(self, databus_buffer: DataBus, end_processes) -> None:

        print("Graphic process started", flush=True)

        ## Shared memory
        self.databus_buffer = databus_buffer
        self.end_processes = end_processes

        # Data bus
        self.databus = DataBus(Point2D(0, 0), Point2D(0, 0), [])

        # Test to plot road and car
        self.iteration = 0

        # Constants
        self.root = tk.Tk()
        self.root.title("Car Racing Board")
        self.root.geometry(str(850) + "x" + str(850))

        folder = 'snake_map_training'
        self.left_side_road_coordinates = pd.read_csv(f'extras/maps/{folder}/road_left.csv')
        self.right_side_road_coordinates = pd.read_csv(f'extras/maps/{folder}/road_right.csv')
        self.fig, self.ax = plt.subplots(figsize=(4, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(column=0, row=11, columnspan=4, rowspan=6, sticky=tk.S, padx=5, pady=5)
        self.root.configure(bg='#0000CC')
        self.root.update()

        ## Main loop
        self.update_display()


    # Update display
    def update_display(self):
        iter = 0

        while not self.end_processes.value:

            if iter % 40 == 0:

                # ---- Empty the buffer ----

                while not self.databus_buffer.empty():
                    self.databus = self.databus_buffer.get()
            
                # ---- Plot interface ----

                # Clear
                self.ax.clear()

                # Road
                self.ax.plot(self.left_side_road_coordinates.x_values, self.left_side_road_coordinates.y_values, color='black', label='left_side')
                self.ax.plot(self.right_side_road_coordinates.x_values, self.right_side_road_coordinates.y_values, color='red', label='right_side')

                # Car
                self.ax.plot(self.databus.car_pos.x, self.databus.car_pos.y, 'bo', label='car')
                self.ax.plot(self.databus.car_ahead_pos.x, self.databus.car_ahead_pos.y, 'ro', label='car2')

                # Sensors
                for detected_point in self.databus.detected_points:
                    self.ax.plot(detected_point.pos.x, detected_point.pos.y, 'go', label='sensor')

                # ---- Draw ----
                self.canvas.draw()
                self.canvas.flush_events()
                self.root.update()

        print("Graphic process correctly stopped", flush=True)
