"""
Base agent class for reinforcement learning agents.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
import torch


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any]):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_step = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select an action given a state.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent with a batch of experience.

        Args:
            batch: Batch of experience data

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's model and parameters.

        Args:
            filepath: Path to save the model
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's model and parameters.

        Args:
            filepath: Path to load the model from
        """
        pass

    def get_training_metrics(self) -> Dict[str, float]:
        """
        Get current training metrics.

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        if self.episode_rewards:
            metrics['mean_episode_reward'] = np.mean(self.episode_rewards[-100:])
            metrics['std_episode_reward'] = np.std(self.episode_rewards[-100:])

        if self.episode_lengths:
            metrics['mean_episode_length'] = np.mean(self.episode_lengths[-100:])

        metrics['training_step'] = self.training_step

        return metrics

    def reset_episode_metrics(self) -> None:
        """Reset episode-level metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
