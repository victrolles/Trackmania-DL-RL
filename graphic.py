from collections import namedtuple
import time

import tkinter as tk

Size_screen = namedtuple('Size_screen', ['width', 'height'])

class Graphic:
    def __init__(self, epsilon, epoch, loss, speed, car_action, time, cancel_training, save_model, end_processes):

        print("Graphic process started", flush=True)

        ## Shared memory

        # Training state
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss

        # Car state
        self.speed = speed
        self.car_action = car_action
        self.time = time

        # Actions
        self.cancel_training = cancel_training
        self.save_model = save_model
        self.end_processes = end_processes

        ## Graphic

        # Constants
        self.screen_size = Size_screen(600, 600)

        # Variables
        self.fps = 0

        # Tkinter
        self.root = tk.Tk()
        self.root.title("Car Racing Board")
        self.root.geometry(str(self.screen_size.width) + "x" + str(self.screen_size.height))
        self.root.resizable(False, False)
        self.root.configure(bg='#0000CC')
        self.root.update()

        ## Main loop
        self.update_display()

    # Init display
    def init_display(self):

        ## infos:

        # Training state
        self.label_epsilon = tk.Label(self.root, text=f"Epsilon: {self.epsilon.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_epoch = tk.Label(self.root, text=f"Epoch: {self.epoch.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_loss = tk.Label(self.root, text=f"Loss: {self.loss.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # Car state
        self.label_speed = tk.Label(self.root, text=f"Speed: {self.speed.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_car_action = tk.Label(self.root, text=f"Action: {self.car_action.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_time = tk.Label(self.root, text=f"Time: {self.time.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # Graphic state
        self.label_fps = tk.Label(self.root, text=f"FPS: {self.fps}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        ## buttons:

        # cancel training:
        self.button_cancel_training = tk.Button(self.root, text="Cancel training", command=self.change_cancel_training, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_cancel_training = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # save model:
        self.button_save_model = tk.Button(self.root, text="Save model", command=self.change_save_model, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_save_model = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # exit:
        self.button_exit = tk.Button(self.root, text="Exit", command=self.exit, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_exit = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        ## placement:

        # infos:
        self.label_epsilon.place(x=10, y=10)
        self.label_epoch.place(x=10, y=40)
        self.label_loss.place(x=10, y=70)

        self.label_speed.place(x=10, y=100)
        self.label_car_action.place(x=10, y=130)
        self.label_time.place(x=10, y=160)

        self.label_fps.place(x=10, y=190)

        # buttons:
        self.button_cancel_training.place(x=10, y=230)
        self.label_button_cancel_training.place(x=200, y=230)

        self.button_save_model.place(x=10, y=280)
        self.label_button_save_model.place(x=200, y=280)

        self.button_exit.place(x=10, y=330)
        self.label_button_exit.place(x=200, y=330)

    # Actions
    def change_cancel_training(self):
        self.cancel_training.value = not self.cancel_training.value
        self.label_button_cancel_training.config(text="On" if self.cancel_training.value else "Off")

    def change_save_model(self):
        self.save_model.value = not self.save_model.value
        self.label_button_save_model.config(text="On" if self.save_model.value else "Off")

    def exit(self):
        self.end_processes.value = True
        self.root.quit()

    # Update display
    def update_display(self):
        self.init_display()
        iter = 0

        while not self.end_processes.value:

            start = time.perf_counter()
            self.update_infos()
            self.root.update()
            time.sleep(0.1)
            end = time.perf_counter()

            self.fps = int(1 / (end - start))

        print("Graphic process correctly stopped", flush=True)

    # Utils
    def update_infos(self):
        self.label_epsilon.config(text=f"Epsilon: {self.epsilon.value:.3f}")
        self.label_epoch.config(text=f"Epoch: {self.epoch.value}")
        self.label_loss.config(text=f"Loss: {self.loss.value:.3f}")

        self.label_speed.config(text=f"Speed: {self.speed.value}")
        self.label_car_action.config(text=f"Action: {self.car_action.value}")
        self.label_time.config(text=f"Time: {self.time.value}")

        self.label_fps.config(text=f"FPS: {self.fps}")

