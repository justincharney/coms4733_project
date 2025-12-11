"""
DQN Agent implementation for discrete action space grasping.

Based on the milestone document specification:
- MULTIDISCRETE_RESNET architecture
- Epsilon-greedy exploration
- Binary cross-entropy loss
- Replay buffer (FIFO queue, 2000 entries)
- No target network (gamma=0, immediate reward prediction)
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from network import DQNBaseNetwork
from observation_preprocessing import transform_observation, transform_observation_batch


class ReplayBuffer:
    """
    Simple replay buffer for DQN (FIFO queue).
    Based on milestone document: buffer size 2000 entries.
    """

    def __init__(self, capacity=2000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (dict with 'rgb' and 'depth')
            action: Action tuple (pixel_idx, rotation_idx)
            reward: Reward received
            next_state: Next state (dict with 'rgb' and 'depth')
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
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
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


class DQNAgent:
    """
    DQN Agent for discrete grasping tasks.

    Based on milestone document:
    - Epsilon-greedy exploration (1.0 to 0.2 over 8000 steps)
    - Binary cross-entropy loss
    - No target network (gamma=0)
    - Replay buffer (2000 entries)
    """

    def __init__(
        self,
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        pixel_action_dim=40000,
        rotation_action_dim=6,
        lr=0.0005,
        buffer_size=2000,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=8000,
        device=None,
        table_height=0.91,
        depth_noise_std=0.01,
        apply_color_jitter=True,
    ):
        """
        Initialize DQN agent.

        Args:
            rgb_shape: Shape of RGB image
            depth_shape: Shape of depth image
            pixel_action_dim: Number of pixel positions (200*200 = 40000)
            rotation_action_dim: Number of rotation options (6)
            lr: Learning rate (default: 0.0005 as per milestone)
            buffer_size: Replay buffer size (default: 2000 as per milestone)
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay_steps: Steps over which epsilon decays
            device: Device to run on (auto-detect if None)
            table_height: Height of table surface in meters (default: 0.91)
            depth_noise_std: Standard deviation for Gaussian depth noise (default: 0.01)
            apply_color_jitter: Whether to apply color jitter augmentation (default: True)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pixel_action_dim = pixel_action_dim
        self.rotation_action_dim = rotation_action_dim
        self.table_height = table_height
        self.depth_noise_std = depth_noise_std
        self.apply_color_jitter = apply_color_jitter

        # Initialize network
        self.q_network = DQNBaseNetwork(
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
            pixel_action_dim=pixel_action_dim,
            rotation_action_dim=rotation_action_dim,
        ).to(self.device)

        # Optimizer (Adam with learning rate 0.0005 as per milestone)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=0.0)

        # Replay buffer (FIFO queue, 2000 entries as per milestone)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Epsilon-greedy exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.step_count = 0

        # Training statistics
        self.loss_history = []

    def update_epsilon(self):
        """Update epsilon based on linear decay schedule."""
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = (
                self.epsilon_start
                - (self.epsilon_start - self.epsilon_end)
                * (self.step_count / self.epsilon_decay_steps)
            )
        else:
            self.epsilon = self.epsilon_end

    def select_action(self, observation, deterministic=False, action_mask=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            observation: Dict with 'rgb' and 'depth' keys
            deterministic: If True, always use greedy action
            action_mask: Optional boolean mask (200, 200) for valid pixels

        Returns:
            Action tuple (pixel_idx, rotation_idx)
        """
        self.update_epsilon()

        # Epsilon-greedy: random action with probability epsilon
        if not deterministic and np.random.random() < self.epsilon:
            # Random action
            # Resample until valid pixel if action_mask provided
            max_attempts = 100
            for _ in range(max_attempts):
                pixel_idx = np.random.randint(0, self.pixel_action_dim)
                rotation_idx = np.random.randint(0, self.rotation_action_dim)

                # Check if pixel is valid (on table)
                if action_mask is not None:
                    row = pixel_idx // 200
                    col = pixel_idx % 200
                    if action_mask[row, col]:
                        return (pixel_idx, rotation_idx)
                else:
                    return (pixel_idx, rotation_idx)

            # Fallback: return random action even if not on table
            return (pixel_idx, rotation_idx)

        # Greedy action: use Q-network
        # Apply observation preprocessing (transform_observation)
        rgb, depth = transform_observation(
            observation['rgb'],
            observation['depth'],
            table_height=self.table_height,
            depth_noise_std=0.0 if deterministic else self.depth_noise_std,
            apply_color_jitter=self.apply_color_jitter and not deterministic,
            training=not deterministic,
        )

        rgb = rgb.to(self.device)
        depth = depth.to(self.device)

        # Add batch dimension if needed
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

        with torch.no_grad():
            action = self.q_network.get_best_action(rgb, depth, action_mask)

        self.step_count += 1
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.

        Args:
            state: Current state (dict with 'rgb' and 'depth')
            action: Action tuple (pixel_idx, rotation_idx)
            reward: Reward received
            next_state: Next state (dict with 'rgb' and 'depth')
            done: Whether episode terminated
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, batch_size=32):
        """
        Update Q-network using batch from replay buffer.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary with loss information
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, _next_states, _dones = self.replay_buffer.sample(batch_size)

        # Apply observation preprocessing to batch
        rgb_batch, depth_batch = transform_observation_batch(
            [s['rgb'] for s in states],
            [s['depth'] for s in states],
            table_height=self.table_height,
            depth_noise_std=self.depth_noise_std,
            apply_color_jitter=self.apply_color_jitter,
            training=True,
        )
        rgb_batch = rgb_batch.to(self.device)
        depth_batch = depth_batch.to(self.device)

        # Get Q-values for selected actions
        q_values, _ = self.q_network.forward(rgb_batch, depth_batch)

        # Extract Q-values for selected actions
        q_selected = []
        for i, (pixel_idx, rotation_idx) in enumerate(actions):
            row = pixel_idx // 200
            col = pixel_idx % 200
            q_selected.append(q_values[i, row, col, rotation_idx])
        q_selected = torch.stack(q_selected)

        # Convert rewards to binary targets (binary classification: success/failure)
        # Binary cross-entropy requires targets in [0, 1]
        # Reward > 0 means success (1), reward <= 0 means failure (0)
        targets = (torch.from_numpy(rewards) > 0).float().to(self.device)
        # Clamp to ensure targets are exactly in [0, 1] range
        targets = torch.clamp(targets, min=0.0, max=1.0)

        # Binary cross-entropy loss (as per milestone document)
        # Convert Q-values to probabilities using sigmoid
        q_probs = torch.sigmoid(q_selected)
        loss = F.binary_cross_entropy(q_probs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.loss_history.append(loss.item())

        return {
            'loss': loss.item(),
            'q_mean': q_selected.mean().item(),
            'q_std': q_selected.std().item(),
        }

    def save(self, filepath):
        """
        Save agent state to file.

        Args:
            filepath: Path to save file
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'loss_history': self.loss_history,
        }, filepath)

    def load(self, filepath):
        """
        Load agent state from file.

        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.step_count = checkpoint.get('step_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])
