"""
Evaluation metrics for robotic grasping tasks.
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


def compute_success_rate(episodes: List[Dict[str, Any]]) -> float:
    """
    Compute success rate across episodes.

    Args:
        episodes: List of episode data

    Returns:
        Success rate (0.0 to 1.0)
    """
    if not episodes:
        return 0.0

    successes = sum(1 for ep in episodes if ep.get('success', False))
    return successes / len(episodes)


def compute_average_reward(episodes: List[Dict[str, Any]]) -> float:
    """
    Compute average reward across episodes.

    Args:
        episodes: List of episode data

    Returns:
        Average reward
    """
    if not episodes:
        return 0.0

    rewards = [ep.get('total_reward', 0) for ep in episodes]
    return np.mean(rewards)


def compute_average_episode_length(episodes: List[Dict[str, Any]]) -> float:
    """
    Compute average episode length.

    Args:
        episodes: List of episode data

    Returns:
        Average episode length
    """
    if not episodes:
        return 0.0

    lengths = [ep.get('episode_length', 0) for ep in episodes]
    return np.mean(lengths)


def compute_grasp_efficiency(
    episodes: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute grasp efficiency metrics.

    Args:
        episodes: List of episode data

    Returns:
        Dictionary of efficiency metrics
    """
    if not episodes:
        return {'grasp_attempts': 0, 'successful_grasps': 0, 'efficiency': 0.0}

    grasp_attempts = 0
    successful_grasps = 0

    for episode in episodes:
        actions = episode.get('actions', [])
        success = episode.get('success', False)

        # Count grasp attempts (gripper closing actions)
        # assuming action 6 is gripper close
        grasp_actions = sum(1 for action in actions if action == 6)
        grasp_attempts += grasp_actions

        if success:
            successful_grasps += 1

    efficiency = successful_grasps / max(grasp_attempts, 1)

    return {
        'grasp_attempts': grasp_attempts,
        'successful_grasps': successful_grasps,
        'efficiency': efficiency
    }


def compute_learning_curves(
    episodes: List[Dict[str, Any]],
    window_size: int = 100
) -> Dict[str, List[float]]:
    """
    Compute learning curves with moving averages.

    Args:
        episodes: List of episode data
        window_size: Window size for moving average

    Returns:
        Dictionary of learning curves
    """
    if not episodes:
        return {'rewards': [], 'success_rates': [], 'lengths': []}

    # Sort episodes by episode number if available
    if 'episode_number' in episodes[0]:
        episodes = sorted(episodes, key=lambda x: x['episode_number'])

    rewards = [ep.get('total_reward', 0) for ep in episodes]
    successes = [ep.get('success', False) for ep in episodes]
    lengths = [ep.get('episode_length', 0) for ep in episodes]

    # Compute moving averages
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    curves = {
        'rewards': moving_average(rewards, window_size),
        'success_rates': moving_average(successes, window_size),
        'lengths': moving_average(lengths, window_size)
    }

    return curves


def compute_action_statistics(
    episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute statistics about action usage.

    Args:
        episodes: List of episode data

    Returns:
        Dictionary of action statistics
    """
    if not episodes:
        return {}

    all_actions = []
    for episode in episodes:
        all_actions.extend(episode.get('actions', []))

    if not all_actions:
        return {}

    action_counts = defaultdict(int)
    for action in all_actions:
        action_counts[action] += 1

    total_actions = len(all_actions)
    action_proportions = {
        action: count/total_actions for action, count in action_counts.items()
        }

    return {
        'action_counts': dict(action_counts),
        'action_proportions': action_proportions,
        'total_actions': total_actions,
        'unique_actions': len(action_counts)
    }


def compute_convergence_metrics(
    episodes: List[Dict[str, Any]],
    convergence_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Compute convergence metrics for training.

    Args:
        episodes: List of episode data
        convergence_threshold: Success rate threshold for convergence

    Returns:
        Dictionary of convergence metrics
    """
    if not episodes:
        return {'converged': False, 'convergence_episode': None}

    # Sort episodes by episode number if available
    if 'episode_number' in episodes[0]:
        episodes = sorted(episodes, key=lambda x: x['episode_number'])

    # Compute success rates in windows
    window_size = 50
    success_rates = []

    for i in range(window_size, len(episodes) + 1):
        window_episodes = episodes[i-window_size:i]
        success_rate = compute_success_rate(window_episodes)
        success_rates.append(success_rate)

    # Check for convergence
    converged = any(rate >= convergence_threshold for rate in success_rates)
    convergence_episode = None

    if converged:
        for i, rate in enumerate(success_rates):
            if rate >= convergence_threshold:
                convergence_episode = i + window_size
                break

    return {
        'converged': converged,
        'convergence_episode': convergence_episode,
        'final_success_rate': success_rates[-1] if success_rates else 0.0,
        'max_success_rate': max(success_rates) if success_rates else 0.0
    }


def compute_comprehensive_metrics(
    episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        episodes: List of episode data

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'success_rate': compute_success_rate(episodes),
        'average_reward': compute_average_reward(episodes),
        'average_length': compute_average_episode_length(episodes),
        'grasp_efficiency': compute_grasp_efficiency(episodes),
        'action_statistics': compute_action_statistics(episodes),
        'convergence_metrics': compute_convergence_metrics(episodes),
        'learning_curves': compute_learning_curves(episodes)
    }

    return metrics
