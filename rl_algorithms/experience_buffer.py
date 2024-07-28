from collections import deque
import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def _append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = [self.buffer[idx].state for idx in indices]
        actions = [self.buffer[idx].action for idx in indices]
        rewards = [self.buffer[idx].reward for idx in indices]
        dones = [self.buffer[idx].done for idx in indices]
        next_states = [self.buffer[idx].next_state for idx in indices]
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(dones, dtype=np.uint8), 
                np.array(next_states))