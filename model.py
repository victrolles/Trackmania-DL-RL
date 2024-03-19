from torch import nn
import torch
import torch.nn.functional as F

from config import HIDDEN_DIM

# The policy model will output actions given states.
# For continuous action spaces, it often outputs parameters of a probability distribution (e.g., a Gaussian distribution)
# from which actions are sampled.
class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(PolicyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Output logits for each action
        )

    def forward(self, state):
        return self.network(state)  # Return logits directly
    
# You'll need two Q-models for the SAC algorithm.
# These models estimate the value of taking particular actions in given states.
class QModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(QModel, self).__init__()
        # Input layer size depends on whether actions are one-hot encoded or represented as indices
        # For one-hot encoding, the input size to the first layer would be state_dim + action_dim
        # For simplicity, we'll design for one-hot encoded actions; adjust if using indices
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single Q-value for the given state-action pair
        )

    def forward(self, state, action):
        # Assuming action is already one-hot encoded; if not, encode it here
        if action.dim() == 1 or action.size(1) == 1:  # Action is provided as indices
            action = F.one_hot(action.long(), num_classes=self.network[0].in_features - self.state_dim).float()

        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        return self.network(x)