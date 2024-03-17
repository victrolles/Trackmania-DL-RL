from collections import namedtuple
import time

import tkinter as tk

Size_screen = namedtuple('Size_screen', ['width', 'height'])

class Graphic:
    def __init__(self, epsilon, epoch, loss, best_dist, current_dist, buffer_size, speed, car_action, time, is_training_mode, is_model_saved, game_speed, end_processes):

        print("Graphic process started", flush=True)

        ## Shared memory

        # Training state
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss
        self.best_dist = best_dist
        self.current_dist = current_dist
        self.buffer_size = buffer_size

        # Car state
        self.speed = speed
        self.car_action = car_action
        self.time = time

        # Actions
        self.is_training_mode = is_training_mode
        self.is_model_saved = is_model_saved
        self.game_speed = game_speed
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
        self.label_best_dist = tk.Label(self.root, text=f"Best dist: {self.best_dist.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_current_dist = tk.Label(self.root, text=f"Current dist: {self.current_dist.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_buffer_size = tk.Label(self.root, text=f"Buffer size: {self.buffer_size.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # Car state
        self.label_speed = tk.Label(self.root, text=f"Speed: {self.speed.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_car_action = tk.Label(self.root, text=f"Action: {self.car_action.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_time = tk.Label(self.root, text=f"Time: {self.time.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # Graphic state
        self.label_fps = tk.Label(self.root, text=f"FPS: {self.fps}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # cancel training:
        self.label_button_is_training_mode = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # save model:
        self.label_button_is_model_saved = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # exit:
        self.label_button_exit = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        # game speed:
        self.label_game_speed = tk.Label(self.root, text="Game speed:", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        ## buttons:

        # cancel training:
        self.button_is_training_mode = tk.Button(self.root, text="Cancel training", command=self.switch_mode, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

        # save model:
        self.button_is_model_saved = tk.Button(self.root, text="Save model", command=self.save_model, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

        # exit:
        self.button_exit = tk.Button(self.root, text="Exit", command=self.exit, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

        # game speed:
        self.button_game_speed = tk.Button(self.root, text="Set", command=self.change_game_speed, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

        ## Entry

        # game speed:
        self.entry_game_speed = tk.Entry(self.root, bg="#FFFFFF", fg="#000000", font=("Arial", 15))

        ## placement:

        # infos:
        self.label_epsilon.place(x=10, y=10)
        self.label_epoch.place(x=10, y=40)
        self.label_loss.place(x=10, y=70)
        self.label_best_dist.place(x=10, y=100)
        self.label_current_dist.place(x=200, y=100)
        self.label_buffer_size.place(x=200, y=70)

        self.label_speed.place(x=10, y=130)
        self.label_car_action.place(x=10, y=160)
        self.label_time.place(x=10, y=190)

        self.label_fps.place(x=200, y=190)

        # buttons:
        self.button_is_training_mode.place(x=10, y=230)
        self.label_button_is_training_mode.place(x=200, y=230)

        self.button_is_model_saved.place(x=10, y=280)
        self.label_button_is_model_saved.place(x=200, y=280)

        self.button_exit.place(x=10, y=330)
        self.label_button_exit.place(x=200, y=330)

    ## Actions
    
    # Switch between training and testing mode
    def switch_mode(self):
        self.is_training_mode.value = not self.is_training_mode.value
        self.label_button_is_training_mode.config(text="Mode : Training" if self.is_training_mode.value else "Mode : Testing")

    def save_model(self):
        self.is_model_saved.value = not self.is_model_saved.value
        self.label_button_is_model_saved.config(text="Saving models ..." if self.is_model_saved.value else "Save models")

    def change_game_speed(self):
        self.game_speed.value = float(self.entry_game_speed.get())
        self.entry_game_speed.delete(0, 'end')
        self.label_game_speed.config(text=f"Game speed: {self.game_speed.value}")

    def exit(self):
        self.end_processes.value = True
        self.root.quit()

    # Update display
    def update_display(self):
        self.init_display()

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
        self.label_best_dist.config(text=f"Best dist: {self.best_dist.value:.3f}")
        self.label_current_dist.config(text=f"Current dist: {self.current_dist.value:.3f}")
        self.label_buffer_size.config(text=f"Buffer size: {self.buffer_size.value}")

        self.label_speed.config(text=f"Speed: {self.speed.value}")
        self.label_car_action.config(text=f"Action: {self.car_action.value}")
        self.label_time.config(text=f"Time: {self.time.value}")

        self.label_fps.config(text=f"FPS: {self.fps}")

