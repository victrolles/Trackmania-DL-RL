# from collections import namedtuple
# import time

# import tkinter as tk
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# from utils import convert_seconds

# Size_screen = namedtuple('Size_screen', ['width', 'height'])

# class Graphic:
#     def __init__(self, epsilon, epoch, loss, best_dist, step, reward, training_time, speed, car_action, game_time, current_dist, is_training_mode, is_model_saved, game_speed, end_processes):

#         print("Graphic process started", flush=True)

#         ## Shared memory

#         # Training state
#         self.epsilon = epsilon
#         self.step = step
#         self.epoch = epoch
#         self.loss = loss
#         self.best_dist = best_dist
#         self.current_dist = current_dist
#         self.reward = reward
#         self.training_time = training_time

#         # Car state
#         self.speed = speed
#         self.car_action = car_action
#         self.game_time = game_time

#         # Actions
#         self.is_training_mode = is_training_mode
#         self.is_model_saved = is_model_saved
#         self.game_speed = game_speed
#         self.end_processes = end_processes

#         ## Graphic

#         # Constants
#         self.screen_size = Size_screen(850, 850)

#         # Variables
#         self.fps = 0
#         self.start_training_time = time.time()

#         # Tkinter
#         self.root = tk.Tk()
#         self.root.title("Car Racing Board")
#         self.root.geometry(str(self.screen_size.width) + "x" + str(self.screen_size.height))

#         # Matplotlib
#         self.plot = Plot(self.root, self.epoch, self.loss, self.current_dist)

#         # Init
#         self.root.resizable(False, False)
#         self.root.configure(bg='#0000CC')
#         self.root.update()

        

#         ## Main loop
#         self.update_display()

#     # Init display
#     def init_display(self):

#         ## labels decoration:

#         self.label_title = tk.Label(self.root, text="Trackmania Training Board", bg="#0000CC", fg="#FFFFFF", font=("Arial", 20))
#         self.label_training_part = tk.Label(self.root, text="Training Informations", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_car_part = tk.Label(self.root, text="Car state", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_action_part = tk.Label(self.root, text="Actions", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         ## infos:

#         # Training state
#         # self.label_episode = tk.Label(self.root, text=f"Episode: {self.episode.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_epsilon = tk.Label(self.root, text=f"Epsilon: {self.epsilon.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_step = tk.Label(self.root, text=f"Step: {self.step.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_best_dist = tk.Label(self.root, text=f"Best dist: {self.best_dist.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_current_dist = tk.Label(self.root, text=f"Current dist: {self.current_dist.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_reward = tk.Label(self.root, text=f"Total reward: {self.reward.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_training_time = tk.Label(self.root, text=f"Training time: {self.training_time.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         # Car state
#         self.label_speed = tk.Label(self.root, text=f"Speed: {self.speed.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_car_action = tk.Label(self.root, text=f"Action: {self.car_action.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
#         self.label_game_time = tk.Label(self.root, text=f"Time: {self.game_time.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         # Graphic state
#         self.label_fps = tk.Label(self.root, text=f"FPS: {self.fps}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         # cancel training:
#         self.label_button_is_training_mode = tk.Label(self.root, text="Mode : Training", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         # save model:
#         self.label_button_is_model_saved = tk.Label(self.root, text="Save models", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         # game speed:
#         self.label_game_speed = tk.Label(self.root, text="Game speed : 1.0", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

#         ## buttons:

#         # cancel training:
#         self.button_is_training_mode = tk.Button(self.root, text="Cancel training", command=self.switch_mode, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

#         # save model:
#         self.button_is_model_saved = tk.Button(self.root, text="Save model", command=self.save_model, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

#         # exit:
#         self.button_exit = tk.Button(self.root, text="Exit", command=self.exit, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

#         # game speed:
#         self.button_game_speed = tk.Button(self.root, text="Set", command=self.change_game_speed, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

#         ## Entry

#         # game speed:
#         self.entry_game_speed = tk.Entry(self.root, bg="#FFFFFF", fg="#000000", font=("Arial", 15), width=8)

#         ## placement:

#         # configure the grid
#         self.root.columnconfigure(0, weight=2)
#         self.root.columnconfigure(1, weight=2)
#         self.root.columnconfigure(2, weight=1)
#         self.root.columnconfigure(3, weight=1)

#         # labels decoration:
#         self.label_title.grid(column=0, row=0, columnspan=4, sticky=tk.N, padx=5, pady=5)
#         self.label_training_part.grid(column=0, row=1, sticky=tk.N, padx=5, pady=5)
#         self.label_car_part.grid(column=1, row=1, sticky=tk.N, padx=5, pady=5)
#         self.label_action_part.grid(column=2, row=1, columnspan=2, sticky=tk.N, padx=5, pady=5)

#         # infos:
#         self.label_step.grid(column=0, row=2, sticky=tk.N, padx=5, pady=5)
#         self.label_best_dist.grid(column=0, row=3, sticky=tk.N, padx=5, pady=5)
#         self.label_epsilon.grid(column=0, row=5, sticky=tk.N, padx=5, pady=5)
#         self.label_reward.grid(column=0, row=6, sticky=tk.N, padx=5, pady=5)
#         self.label_training_time.grid(column=0, row=7, sticky=tk.N, padx=5, pady=5)

#         self.label_game_time.grid(column=1, row=2, sticky=tk.N, padx=5, pady=5)
#         self.label_current_dist.grid(column=1, row=3, sticky=tk.N, padx=5, pady=5)
#         self.label_speed.grid(column=1, row=4, sticky=tk.N, padx=5, pady=5)
#         self.label_car_action.grid(column=1, row=5, sticky=tk.N, padx=5, pady=5)
        

#         self.label_fps.grid(column=2, row=2, columnspan=2, sticky=tk.N, padx=5, pady=5)

#         # buttons:
#         self.button_is_training_mode.grid(column=2, row=4, sticky=tk.E, padx=5, pady=5)
#         self.label_button_is_training_mode.grid(column=3, row=4, sticky=tk.N, padx=5, pady=5)

#         self.button_is_model_saved.grid(column=2, row=5, sticky=tk.E, padx=5, pady=5)
#         self.label_button_is_model_saved.grid(column=3, row=5, sticky=tk.N, padx=5, pady=5)

#         self.button_exit.grid(column=0, row=8, sticky=tk.N, padx=5, pady=5)

#         self.entry_game_speed.grid(column=2, row=7, sticky=tk.E, padx=10, pady=0)
#         self.button_game_speed.grid(column=3, row=7, sticky=tk.W, padx=10, pady=5)     
#         self.label_game_speed.grid(column=2, row=8, columnspan=2, sticky=tk.N, padx=5, pady=5)

#     ## Actions
    
#     # Switch between training and testing mode
#     def switch_mode(self):
#         self.is_training_mode.value = not self.is_training_mode.value
#         self.label_button_is_training_mode.config(text="Mode : Training" if self.is_training_mode.value else "Mode : Testing")

#     def save_model(self):
#         self.is_model_saved.value = not self.is_model_saved.value
#         self.label_button_is_model_saved.config(text="Saving models ..." if self.is_model_saved.value else "Save models")

#     def check_save_model(self):
#         if self.is_model_saved.value:
#             self.label_button_is_model_saved.config(text="Saving models ...")
#         else:
#             self.label_button_is_model_saved.config(text="Save models")

#     def change_game_speed(self):
#         self.game_speed.value = float(self.entry_game_speed.get())
#         self.entry_game_speed.delete(0, 'end')
#         self.label_game_speed.config(text=f"Game speed : {self.game_speed.value}")

#     def exit(self):
#         self.end_processes.value = True
#         self.root.quit()

#     # Update display
#     def update_display(self):
#         self.init_display()
#         iter = 0

#         while not self.end_processes.value:

#             start = time.perf_counter()

#             if iter % 10 == 0:
#                 self.update_infos()

#             if iter % 20 == 0:
#                 self.check_save_model()
            
#             if iter % 10 == 0:
#                 self.plot.update_infos()

#             if iter % 20 == 0:
#                 self.plot.update_plot()

#             if iter % 10 == 0:  
#                 self.root.minsize(self.screen_size.width, self.screen_size.height)
#                 self.root.maxsize(self.screen_size.width, self.screen_size.height)
#                 self.root.update()
            
#             time.sleep(0.01)
#             end = time.perf_counter()

#             self.fps = int(1 / (end - start)) / 10
#             iter += 1
#             if iter > 1000:
#                 iter = 0

#         print("Graphic process correctly stopped", flush=True)

#     # Utils
#     def update_infos(self):
#         self.label_epsilon.config(text=f"Episode: {self.epsilon.value:.3f}")
#         self.label_step.config(text=f"Step: {self.step.value}")
#         self.label_best_dist.config(text=f"Best dist: {self.best_dist.value:.3f}")
#         self.label_current_dist.config(text=f"Current dist: {self.current_dist.value:.3f}")
#         self.label_reward.config(text=f"Total reward: {self.reward.value}")
        
#         self.training_time.value = int(time.time() - self.start_training_time)
#         hours, minutes, seconds = convert_seconds(self.training_time.value)
#         self.label_training_time.config(text=f"Training time: {hours} h {minutes} m {seconds} s")

#         self.label_speed.config(text=f"Speed: {self.speed.value}")
#         self.label_car_action.config(text=f"Action: {self.car_action.value}")
#         self.label_game_time.config(text=f"Time: {self.game_time.value}")

#         self.label_fps.config(text=f"FPS: {self.fps}")

# class Plot:
#     def __init__(self, root, epoch, loss, distance):
#         self.root = root
#         self.epoch = epoch
#         self.loss = loss
#         self.distance = distance

#         self.fig, (self.graph_distances, self.graph_losses) = plt.subplots(1, 2, figsize=(8, 4))
#         self.fig.suptitle('Training Curves')

#         self.list_distances = [0]
#         self.list_losses = [0.0]

#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
#         self.canvas.get_tk_widget().grid(column=0, row=11, columnspan=4, rowspan=6, sticky=tk.S, padx=5, pady=5)

#     def update_infos(self):
#         if self.loss.value != self.list_losses[-1]:
#             self.list_losses.append(self.loss.value)
#             if self.distance.value != self.list_distances[-1] and self.distance.value > 3.0:
#                 self.list_distances.append(self.distance.value)

#     def update_plot(self):
#         # Update graph distances
#         self.graph_distances.clear()
#         self.graph_distances.set_title('Distances :')
#         self.graph_distances.set_xlabel('Tries')
#         self.graph_distances.set_ylabel('Distance')
#         self.graph_distances.plot(self.list_distances)
#         self.graph_distances.set_ylim(ymin=0)
#         self.graph_distances.text(len(self.list_distances)-1, self.list_distances[-1], str(self.list_distances[-1]))

#         # Update graph losses
#         self.graph_losses.clear()
#         self.graph_losses.set_title('Losses :')
#         self.graph_losses.set_xlabel('Epochs')
#         self.graph_losses.set_ylabel('Loss')
#         self.graph_losses.plot(self.list_losses)
#         self.graph_losses.set_ylim(ymin=0)
#         self.graph_losses.text(len(self.list_losses)-1, self.list_losses[-1], "{:.2f}".format(self.list_losses[-1]))

#         # Draw
#         self.canvas.draw()
#         self.canvas.flush_events()

# # class Plot:
# #     def __init__(self, root, policy_loss, q1_loss, q2_loss, distance):
# #         self.root = root
# #         self.policy_loss = policy_loss
# #         self.q1_loss = q1_loss
# #         self.q2_loss = q2_loss
# #         self.distance = distance

# #         self.fig, (self.graph_distances, self.graph_policy_loss) = plt.subplots(1, 2, figsize=(8, 4))
# #         self.fig.suptitle('Training Curves')
# #         self.graph_q_losses = self.graph_policy_loss.twinx()

# #         self.list_distances = [0]
# #         self.list_policy_losses = [0.0]
# #         self.list_q1_losses = [0.0]
# #         self.list_q2_losses = [0.0]

# #         self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
# #         self.canvas.get_tk_widget().grid(column=0, row=11, columnspan=4, rowspan=6, sticky=tk.S, padx=5, pady=5)

# #     def update_infos(self):
# #         if self.policy_loss.value != self.list_policy_losses[-1]:
# #             self.list_policy_losses.append(self.policy_loss.value)

# #         if self.q1_loss.value != self.list_q1_losses[-1]:
# #             self.list_q1_losses.append(self.q1_loss.value)
# #             if self.distance.value != self.list_distances[-1] and self.distance.value > 3.0:
# #                 self.list_distances.append(self.distance.value)

# #         if self.q2_loss.value != self.list_q2_losses[-1]:
# #             self.list_q2_losses.append(self.q2_loss.value)

# #     def update_plot(self):
# #         # Update graph distances
# #         self.graph_distances.clear()
# #         self.graph_distances.set_title('Distances :')
# #         self.graph_distances.set_xlabel('Tries')
# #         self.graph_distances.set_ylabel('Distance')
# #         self.graph_distances.plot(self.list_distances)
# #         # self.graph_distances.set_ylim(ymin=0)
# #         self.graph_distances.text(len(self.list_distances)-1, self.list_distances[-1], str(self.list_distances[-1]))

# #         # Update graph losses
# #         self.graph_policy_loss.clear()
# #         self.graph_policy_loss.set_title('Losses :')
# #         self.graph_policy_loss.set_xlabel('Epochs')
# #         self.graph_policy_loss.set_ylabel('Loss')
# #         self.graph_policy_loss.plot(self.list_policy_losses, label="Policy loss", color="blue")
# #         self.graph_policy_loss.legend(loc='upper left')

# #         self.graph_q_losses.clear()
# #         self.graph_q_losses.plot(self.list_q1_losses, label="Q1 loss", color="green")
# #         self.graph_q_losses.plot(self.list_q2_losses, label="Q2 loss", color="red")
# #         self.graph_q_losses.legend(loc='upper right')


# #         # Draw
# #         self.canvas.draw()
# #         self.canvas.flush_events()