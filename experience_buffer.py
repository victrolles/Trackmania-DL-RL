from collections import namedtuple
from collections import deque
import numpy as np

from config import BUFFER_SIZE, BATCH_SIZE

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done', 'next_state'))

class ExperienceBuffer:
    def __init__(self):
        self.buffer = deque(maxlen = BUFFER_SIZE)

    def __len__(self):
        return len(self.buffer)

    def _append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)