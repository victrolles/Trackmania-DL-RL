import torch
import torch.nn as nn
import torch.optim as optim

from experience_buffer import Experience
from config import LR, GAMMA, BUFFER_SIZE, SYNC_MODELS

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

    def update_model(self):
        if len(self.experience_buffer) < BUFFER_SIZE:
            return

        transitions = self.experience_buffer.sample(BUFFER_SIZE)
        batch = Experience(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()

    def sync_target_network(self):
        if self.epoch.value % SYNC_MODELS == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # def save_model(self):
    #     torch.save({
    #         'model_network_state_dict': self.model_network.state_dict(),
    #         'model_target_network_state_dict': self.model_target_network.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'best_score': self.best_score.value,
    #     }, 'model.pth')

    # def load_model(self):
    #     checkpoint = torch.load('model.pth')
    #     self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
    #     self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.best_score.value = checkpoint['best_score']

    #     self.model_network.eval()
    #     self.model_target_network.eval()