import torch
from replay_memory import Experience

class DQNModelOptimiser:

    def __init__(self, model, target_model, replay_memory, batch_size, gamma, optimizer, loss_function, device):
        self.model = model
        self.target_model = target_model
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def optimise_model(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.replay_memory.sample(self.batch_size)
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
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()