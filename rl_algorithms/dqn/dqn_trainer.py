import torch
import torch.nn as nn
import torch.optim as optim

from config.globals import LR, GAMMA, BATCH_SIZE, SYNC_TARGET_RATE, SAVE_MODELS_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY, LOAD_SAVED_MODEL, TRACK_NAME
from rl_algorithms.dqn.dqn_model import DQNModel
from rl_algorithms.experience_buffer import ExperienceBuffer

class DQNTrainer:

    def __init__(self, experience_buffer: ExperienceBuffer, device):

        # Device
        self.device = device
        self.stop_training = False
        
        # Model
        self.model_network = DQNModel(5, 256, 5).to(self.device)
        self.model_target_network = DQNModel(5, 256, 5).to(self.device)

        if LOAD_SAVED_MODEL:
            self.load_model()

        # Experience buffer : shared memory
        self.experience_buffer = experience_buffer

        # Training states
        self.epsilon = 0
        self.epoch = 0
        self.step = 0
        self.loss = 0
        self.training_time = 0

        # Training parameters
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def train_model(self):
        
        self.epoch += 1
        self.step = 0

        while len(self.experience_buffer) > BATCH_SIZE:

            self.step += 1

            self.update_model()
            
            self.update_epsilon()

            self.sync_target_network()

            if self.stop_training:
                break

    def update_model(self):

        self.optimizer.zero_grad()

        if len(self.experience_buffer) < BATCH_SIZE:
            batch = self.experience_buffer.sample(len(self.experience_buffer))
        else:
            batch = self.experience_buffer.sample(BATCH_SIZE)
        
        states, actions, rewards, dones, next_states = batch

        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.int, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        dones = torch.ByteTensor(dones, device=self.device)

        state_action_values = self.model_network(states).gather(1, torch.argmax(actions, dim=1).unsqueeze(1)).squeeze(1)
        next_state_values = self.model_target_network(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        # Compute the expected state action values
        expected_state_action_values = next_state_values * GAMMA + rewards

        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss = loss.item()

        # Update the model
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, EPSILON_START - self.epoch * EPSILON_DECAY)

    def sync_target_network(self):
        if self.epoch % SYNC_TARGET_RATE == 0:
            self.model_target_network.load_state_dict(self.model_network.state_dict())

    def save_model(self):
        if self.epoch % SAVE_MODELS_RATE == 0:
            torch.save({
                'model_network_state_dict': self.model_network.state_dict(),
                'model_target_network_state_dict': self.model_target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'trainig_time': self.training_time
            }, f'extras/maps/{TRACK_NAME}/saves/model.pth')
            print("Models correctly saved")

    def load_model(self):
        checkpoint = torch.load(f'extras/maps/{TRACK_NAME}/saves/model.pth')
        self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
        self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.training_time = checkpoint['trainig_time']

        self.model_network.eval()
        self.model_target_network.eval()

        print("Models correctly loaded")