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
        sampled_experiences = [self.buffer[idx] for idx in indices]
        
        # Remove the sampled experiences from the buffer
        for idx in sorted(indices, reverse=True):
            del self.buffer[idx]

        states = [exp.state for exp in sampled_experiences]
        actions = [exp.action for exp in sampled_experiences]
        rewards = [exp.reward for exp in sampled_experiences]
        dones = [exp.done for exp in sampled_experiences]
        next_states = [exp.next_state for exp in sampled_experiences]
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(dones, dtype=np.uint8), 
                np.array(next_states))