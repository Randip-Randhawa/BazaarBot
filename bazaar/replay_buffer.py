import numpy as np

from bazaar.env import OBS_SIZE


class ReplayBuffer:
    """Fixed-size memory of past transitions. Once full, new ones overwrite the oldest."""

    def __init__(self, capacity, seed=None):
        self.capacity = capacity
        self.states = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

        self.next_slot = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated):
        # Assigning into the arrays copies the values, so the caller can
        # safely reuse or modify its own observation arrays afterwards.
        i = self.next_slot
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = terminated

        self.next_slot = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Pick a random batch of distinct stored transitions, in DQNAgent.update() order."""
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminated[idx],
        )
