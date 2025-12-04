import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Basic replay buffer for storing and sampling transitions.
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (dict observation)
            action: Action taken
            reward: Reward received
            next_state: Next state (dict observation)
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        rng = np.random.default_rng()
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states = [transition[0] for transition in batch]
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch], dtype=np.float32)
        next_states = [transition[3] for transition in batch]
        dones = np.array([transition[4] for transition in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
