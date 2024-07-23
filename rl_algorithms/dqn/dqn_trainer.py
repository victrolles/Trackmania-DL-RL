import torch
import torch.nn as nn
import torch.optim as optim
from dqn_model import DQNModel

from config import LR, GAMMA, BATCH_SIZE, SYNC_MODELS_RATE, SAVE_MODELS_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY, LOAD_SAVED_MODEL

class DQNTrainer:

    def __init__(self, experience_buffer, epsilon, epoch, loss, device, is_model_saved, end_processes, track_name, training_time):

        # Track name
        self.track_name = track_name
        
        # Model
        self.model_network = DQNModel(5, 256).to(self.device) #400, 512, 3
        self.model_target_network = DQNModel(5, 256).to(self.device) #400, 512, 3

        if LOAD_SAVED_MODEL:
            self.load_model()

        # Experience buffer
        self.experience_buffer = experience_buffer

        # Training states : shared memory
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss
        self.is_model_saved = is_model_saved
        self.end_processes = end_processes
        self.training_time = training_time

        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # Device
        self.device = device

    def train_model(self):
        
        while len(self.experience_buffer) > BATCH_SIZE:

            # Update model
            self.update_model()

            # Update epsilon and epoch
            self.epoch.value += 1
            self.epsilon.value = max(EPSILON_END, EPSILON_START - self.epoch.value * EPSILON_DECAY)

            # Sync models
            if self.epoch.value % SYNC_MODELS_RATE == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Save model
            if self.epoch.value % SAVE_MODELS_RATE == 0 or self.end_processes.value == True:
                self.save_model()

            # End processes
            if self.end_processes.value:
                break

    def update_model(self):
        if len(self.experience_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, dones, next_states = self.experience_buffer.sample()

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.int, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        dones = torch.ByteTensor(dones, device=self.device)

        state_action_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_model(next_states).max(1)[0].detach()

        # Compute the expected state action values
        expected_state_action_values = next_state_values * GAMMA + rewards

        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        # print(f'loss: {loss.item()}')
        self.loss.value = loss.item()

        # Update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # exit()

    def save_model(self):
        torch.save({
            'model_network_state_dict': self.model.state_dict(),
            'model_target_network_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch.value,
            'trainig_time': self.training_time.value
        }, f'maps/{self.track_name}/saves/model.pth')

    def load_model(self):
        checkpoint = torch.load(f'maps/{self.track_name}/saves/model.pth')
        self.model.load_state_dict(checkpoint['model_network_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch.value = checkpoint['epoch']
        self.training_time.value = checkpoint['trainig_time']

        self.model.eval()
        self.target_model.eval()