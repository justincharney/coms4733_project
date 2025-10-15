"""
Data processing utilities for the robotic grasping project.
"""

import numpy as np
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
import os


def save_episode_data(
    episode_data: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save episode data to file.

    Args:
        episode_data: Dictionary containing episode data
        filepath: Path to save the data
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(episode_data, f)


def load_episode_data(filepath: str) -> Dict[str, Any]:
    """
    Load episode data from file.

    Args:
        filepath: Path to the data file

    Returns:
        Dictionary containing episode data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_training_config(
    config: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save training configuration to JSON file.

    Args:
        config: Configuration dictionary
        filepath: Path to save the config
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_training_config(filepath: str) -> Dict[str, Any]:
    """
    Load training configuration from JSON file.

    Args:
        filepath: Path to the config file

    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_observations(
    observations: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize observations using z-score normalization.

    Args:
        observations: Array of observations
        mean: Mean for normalization (computed if None)
        std: Standard deviation for normalization (computed if None)

    Returns:
        Normalized observations, mean, std
    """
    if mean is None:
        mean = np.mean(observations, axis=0)
    if std is None:
        std = np.std(observations, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero

    normalized = (observations - mean) / std
    return normalized, mean, std


def create_episode_summary(
    episode_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a summary of episode data.

    Args:
        episode_data: Raw episode data

    Returns:
        Episode summary dictionary
    """
    summary = {
        'episode_length': len(episode_data.get('rewards', [])),
        'total_reward': sum(episode_data.get('rewards', [])),
        'success': episode_data.get('success', False),
        'final_distance': episode_data.get('final_distance', float('inf')),
        'num_contacts': len(episode_data.get('contacts', [])),
        'actions_taken': episode_data.get('actions', []),
        'q_values': episode_data.get('q_values', [])
    }

    return summary


def batch_episodes(
    episodes: List[Dict[str, Any]],
    batch_size: int
) -> List[List[Dict[str, Any]]]:
    """
    Batch episodes into groups.

    Args:
        episodes: List of episode data
        batch_size: Size of each batch

    Returns:
        List of episode batches
    """
    batches = []
    for i in range(0, len(episodes), batch_size):
        batch = episodes[i:i + batch_size]
        batches.append(batch)
    return batches


def compute_episode_statistics(
    episodes: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute statistics across multiple episodes.

    Args:
        episodes: List of episode data

    Returns:
        Dictionary of statistics
    """
    rewards = [ep.get('total_reward', 0) for ep in episodes]
    lengths = [ep.get('episode_length', 0) for ep in episodes]
    successes = [ep.get('success', False) for ep in episodes]

    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean(successes),
        'num_episodes': len(episodes)
    }

    return stats


def create_training_dataset(
    episodes: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Create a training dataset from episodes.

    Args:
        episodes: List of episode data
        output_path: Path to save the dataset
    """
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'dones': []
    }

    for episode in episodes:
        obs = episode.get('observations', [])
        actions = episode.get('actions', [])
        rewards = episode.get('rewards', [])
        dones = episode.get('dones', [])

        # Add transitions
        for i in range(len(obs) - 1):
            dataset['observations'].append(obs[i])
            dataset['actions'].append(actions[i])
            dataset['rewards'].append(rewards[i])
            dataset['next_observations'].append(obs[i + 1])
            dataset['dones'].append(dones[i])

    # Convert to numpy arrays
    for key in dataset:
        dataset[key] = np.array(dataset[key])

    # Save dataset
    save_episode_data(dataset, output_path)
