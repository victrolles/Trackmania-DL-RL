import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # 1 input channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the feature map after the convolutional layers
        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(256, 8, 4), 4, 2), 3, 1)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(256, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_w * conv_h * 64
        
        self.fc1 = nn.Linear(linear_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

# Example usage:
# state_dim is the shape of the input image (1, 256, 256) for grayscale
# action_dim is the number of possible actions
# model = QModel(action_dim=4)  # Adjust action_dim based on your action space