from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bazaar.env import N_ACTIONS, OBS_SIZE, action_mask


class QNetwork(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ACTIONS),
        )

    def forward(self, obs):
        return self.layers(obs)


class DQNAgent:
    def __init__(self, hidden_size=128, learning_rate=1e-3, gamma=0.99, epsilon=1.0, device=None, seed=None):
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.online_net = QNetwork(hidden_size).to(self.device)
        # A copy of the online network that only changes when update_target()
        # is called, so the TD targets don't shift after every gradient step.
        self.target_net = QNetwork(hidden_size).to(self.device)
        self.update_target()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

    def select_action(self, obs, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if self.rng.random() < epsilon:
            # Explore, but only among actions that make sense right now.
            return int(self.rng.choice(np.flatnonzero(action_mask(obs))))
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        """The valid action with the highest Q-value. Uses no randomness, so
        evaluation can call it without touching the exploration random stream."""
        with torch.no_grad():
            q_values = self.online_net(self._to_tensor(obs, torch.float32))
            q_values[~self._to_tensor(action_mask(obs), torch.bool)] = -torch.inf
        return int(q_values.argmax())

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update(self, states, actions, rewards, next_states, dones):
        """Run one gradient step on a batch of transitions and return the loss.

        `dones` should be the `terminated` flags, not `truncated`: a truncated
        episode didn't really end, so its next state still has value.
        """
        states = self._to_tensor(states, torch.float32)
        actions = self._to_tensor(actions, torch.int64)

        # Q-value of the action that was actually taken in each transition.
        q_taken = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.td_targets(rewards, next_states, dones)

        loss = F.smooth_l1_loss(q_taken, targets)  # Huber loss, less thrown by outliers than MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def td_targets(self, rewards, next_states, dones):
        """r + gamma * max_a' Q_target(s', a'), where the max only considers
        actions that are valid in s', and there is no future value once the
        episode has ended."""
        next_valid = self._to_tensor(action_mask(next_states), torch.bool)
        rewards = self._to_tensor(rewards, torch.float32)
        next_states = self._to_tensor(next_states, torch.float32)
        dones = self._to_tensor(dones, torch.float32)

        with torch.no_grad():
            next_q = self.target_net(next_states).masked_fill(~next_valid, -torch.inf)
            best_next_q = next_q.max(dim=1).values
            return rewards + self.gamma * (1 - dones) * best_next_q

    def save(self, path):
        """Save the trained network, enough to rebuild the agent for evaluation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden_size": self.hidden_size, "weights": self.online_net.state_dict()}, path)

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(hidden_size=checkpoint["hidden_size"], epsilon=0.0, device=device)
        agent.online_net.load_state_dict(checkpoint["weights"])
        agent.update_target()
        return agent

    def _to_tensor(self, values, dtype):
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)
