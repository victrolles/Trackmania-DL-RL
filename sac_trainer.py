from typing import Literal

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from config import LR, GAMMA, ALPHA, LOAD_SAVED_MODEL
from model import PolicyModel, QModel

class SACTrainer:

    def __init__(self,
                 policy_model: PolicyModel,
                 q1_model: QModel,
                 q2_model: QModel,
                 device: Literal['cuda', 'cpu'],
                 is_model_saved: bool,
                 end_processes: bool,
                 track_name: str,
                 episode: int,
                 loss: float,
                 training_time: int):

        # Track name
        self.track_name = track_name

        # Training states : shared memory
        self.episode = episode
        self.loss = loss
        self.is_model_saved = is_model_saved
        self.end_processes = end_processes
        self.training_time = training_time
        
        # Model
        self.policy_model = policy_model
        self.q1_model = q1_model
        self.q2_model = q2_model

        # Device
        self.device = device

        if LOAD_SAVED_MODEL:
            self.load_model()

        # Optimizers
        self.optimizer_policy = optim.Adam(policy_model.parameters(), lr=LR)
        self.optimizer_q1 = optim.Adam(q1_model.parameters(), lr=LR)
        self.optimizer_q2 = optim.Adam(q2_model.parameters(), lr=LR)

    def train_model(self,
                    game_experience: list):

        states, actions, rewards, dones, next_states = zip(*game_experience)

        # Convert to tensors
        states = torch.FloatTensor(states, device=self.device)
        actions = torch.LongTensor(actions, device=self.device).unsqueeze(-1)  # Ensure actions are long for gather
        rewards = torch.FloatTensor(rewards, device=self.device).unsqueeze(-1)
        dones = torch.FloatTensor(dones, device=self.device).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states, device=self.device)

        # Q-models' predictions
        current_q1 = self.q1_model(states).gather(1, actions)  # Select the Q-values of the taken actions
        current_q2 = self.q2_model(states).gather(1, actions)

        # Policy's action logits for next states & softmax for probabilities
        next_action_logits = self.policy_model(next_states)
        next_probabilities = F.softmax(next_action_logits, dim=-1)
        
        # Entropy of the next actions for exploration (assuming a small value for numerical stability)
        next_entropy = -(next_probabilities * torch.log(next_probabilities + 1e-8)).sum(dim=1, keepdim=True)

        # Expected Q-values from the next state, using the policy's probabilities (soft Q-update)
        next_q1 = self.q1_model(next_states)
        next_q2 = self.q2_model(next_states)
        next_q_values = torch.min(next_q1, next_q2)
        expected_q = (next_probabilities * next_q_values).sum(dim=1, keepdim=True) - ALPHA * next_entropy

        # Compute the target for the Q updates
        q_targets = rewards + (GAMMA * (1 - dones) * expected_q)

        # Q-models' losses and updates
        q1_loss = F.mse_loss(current_q1, q_targets.detach())
        q2_loss = F.mse_loss(current_q2, q_targets.detach())
        
        self.optimizer_q1.zero_grad()
        q1_loss.backward()
        self.optimizer_q1.step()
        
        self.optimizer_q2.zero_grad()
        q2_loss.backward()
        self.optimizer_q2.step()

        # Update policy model
        means, log_stds = self.policy_model(states)
        stds = log_stds.exp()
        actions = means + stds * torch.randn_like(stds)

        q1 = self.q1_model(states, actions)
        q2 = self.q2_model(states, actions)
        q_min = torch.min(q1, q2)

        policy_loss = (ALPHA * log_stds.sum(dim=1) - q_min).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update the policy model
        # Calculate the policy model's logits for the current states
        logits = self.policy_model(states)
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Calculate the Q-values for the current states and actions
        q1_values = self.q1_model(states)
        q2_values = self.q2_model(states)
        min_q_values = torch.min(q1_values, q2_values)

        # Calculate the expected Q-values weighted by the action probabilities
        expected_q_values = torch.sum(probabilities * min_q_values, dim=1, keepdim=True)

        # Policy loss is the negative of the expected return plus the entropy bonus
        policy_loss = -(expected_q_values + ALPHA * torch.sum(probabilities * log_probabilities, dim=1, keepdim=True)).mean()

        # Perform backpropagation and update the policy model's parameters
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()



    def save_model(self):
        torch.save({
            'policy_network_state_dict': self.policy_model.state_dict(),
            'q1_network_state_dict': self.q1_model.state_dict(),
            'q2_network_state_dict': self.q2_model.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_q1_state_dict': self.optimizer_q1.state_dict(),
            'optimizer_q2_state_dict': self.optimizer_q2.state_dict(),
            'episode': self.episode.value,
            'training_time': self.training_time.value
        }, f'maps/{self.track_name}/saves/model.pth')

    def load_model(self):
        checkpoint = torch.load(f'maps/{self.track_name}/saves/model.pth')

        self.policy_model.load_state_dict(checkpoint['policy_network_state_dict'])
        self.q1_model.load_state_dict(checkpoint['q1_network_state_dict'])
        self.q2_model.load_state_dict(checkpoint['q2_network_state_dict'])

        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_q1.load_state_dict(checkpoint['optimizer_q1_state_dict'])
        self.optimizer_q2.load_state_dict(checkpoint['optimizer_q2_state_dict'])

        self.episode.value = checkpoint['episode']
        self.training_time.value = checkpoint['training_time']