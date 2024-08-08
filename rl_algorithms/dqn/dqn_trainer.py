import time
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

from librairies.data_classes import TrainingStats, RLConfig

from rl_algorithms.dqn.dqn_model import DQNModel
from rl_algorithms.experience_buffer import ExperienceBuffer

class DQNTrainer:

    def __init__(self, rl_config: RLConfig, experience_buffer: ExperienceBuffer, input_size: int, device) -> None:

        # Device
        self.name = "DQN"
        self.stop_training = False
        self.rl_config = rl_config
        self.device = device
        
        # Model
        self.model_network = DQNModel(input_size, self.rl_config.hidden_layer_size, self.rl_config.output_size).to(self.device)
        self.model_target_network = DQNModel(input_size, self.rl_config.hidden_layer_size, self.rl_config.output_size).to(self.device)

        self.optimizer = optim.Adam(self.model_network.parameters(), lr=self.rl_config.lr)
        self.criterion = nn.MSELoss()

        # if LOAD_SAVED_MODEL:
        #     self.load_model()
            

        # Experience buffer : shared memory
        self.experience_buffer = experience_buffer

        # Training states
        self.epsilon = self.rl_config.
        self.epoch = 0
        self.step = 0
        self.loss_value = 0
        self.training_time = 0

        # # Create a new folder for saving models
        # self.save_dir = f"extras/maps/{TRACK_NAME}/saves/{datetime.datetime.now().strftime("%d-%m-%y-%H-%M")}_{self.name}_{agent_config.name}Agent"
        # os.makedirs(self.save_dir, exist_ok=True)
        # print(f"Created directory: {self.save_dir}")
        

    def train_model(self) -> TrainingStats:
    
        self.epoch += 1
        self.step = 0

        while len(self.experience_buffer) > self.rl_config.batch_size:

            self.step += 1

            self.update_model()
            
            self.update_epsilon()

            self.sync_target_network()

            if self.stop_training:
                break

        return TrainingStats(self.epoch, self.step, self.loss_value, self.epsilon)    

    def update_model(self):

        self.optimizer.zero_grad()

        if len(self.experience_buffer) < self.rl_config.batch_size:
            batch = self.experience_buffer.sample(len(self.experience_buffer))
        else:
            batch = self.experience_buffer.sample(self.rl_config.batch_size)
        
        states, actions, rewards, dones, next_states = batch

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.int, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        state_action_values = self.model_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_state_values = self.model_target_network(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        # Compute the expected state action values
        expected_state_action_values = next_state_values * self.rl_config.gamma + rewards

        # Compute the loss
        self.loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss_value = self.loss.item()

        # Update the model
        self.loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.rl_config.epsilon.end, self.rl_config.epsilon.start - self.epoch * self.rl_config.epsilon.decay)

    def sync_target_network(self):
        if self.epoch % self.rl_config.sync_target_rate == 0:
            self.model_target_network.load_state_dict(self.model_network.state_dict())

    # def save_model(self):
    #     save_path = os.path.join(self.save_dir, f"model_{self.epoch}.pth")
    #     torch.save({
    #         'model_network_state_dict': self.model_network.state_dict(),
    #         'model_target_network_state_dict': self.model_target_network.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'epoch': self.epoch,
    #         'training_time': self.training_time
    #     }, save_path)
    #     print(f"Models correctly saved at epoch {self.epoch}")

    # def load_model(self):
    #     checkpoint = torch.load(f'extras/maps/{TRACK_NAME}/saves/model.pth')
    #     self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
    #     self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.epoch = checkpoint['epoch']
    #     self.training_time = checkpoint['trainig_time']

    #     self.model_network.eval()
    #     self.model_target_network.eval()

    #     print("Models correctly loaded")