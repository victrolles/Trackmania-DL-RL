import torch
import torch.nn as nn
import torch.optim as optim

from config import LR, GAMMA, BATCH_SIZE, SYNC_MODELS_RATE, SAVE_MODELS_RATE, EPSILON_START, EPSILON_END, EPSILON_DECAY

class DQNTrainer:

    def __init__(self, model, target_model, experience_buffer, epsilon, epoch, loss):

        # Model
        self.model = model
        self.target_model = target_model

        # Experience buffer
        self.experience_buffer = experience_buffer

        # Training states : shared memory
        self.epsilon = epsilon
        self.epoch = epoch
        self.loss = loss

        # Training parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_model(self):
        print(f"buffer size: {len(self.experience_buffer)}", flush=True)
        while len(self.experience_buffer) > BATCH_SIZE:

            # Update model
            self.update_model()
            print("Training...")

            # Update epsilon and epoch
            self.epoch.value += 1
            self.epsilon.value = max(EPSILON_END, EPSILON_START - self.epoch.value * EPSILON_DECAY)

            # Sync models
            if self.epoch.value % SYNC_MODELS_RATE == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Save model
            if self.epoch.value % SAVE_MODELS_RATE == 0:
                self.save_model()

    def update_model(self):
        if len(self.experience_buffer) <= BATCH_SIZE:
            return

        states, actions, rewards, dones, next_states = self.experience_buffer.sample()

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.ByteTensor(dones)

        state_action_values = self.model(states).gather(1, torch.argmax(actions, dim=0).unsqueeze(1)).squeeze(1)

        next_state_values = self.target_model(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss.value = loss.item()
        loss.backward()
        self.optimizer.step()
        

    def save_model(self):
        torch.save({
            'model_network_state_dict': self.model.state_dict(),
            'model_target_network_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'model.pth')

    # def load_model(self):
    #     checkpoint = torch.load('model.pth')
    #     self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
    #     self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.best_score.value = checkpoint['best_score']

    #     self.model_network.eval()
    #     self.model_target_network.eval()