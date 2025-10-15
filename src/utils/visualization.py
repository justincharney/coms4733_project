"""
Visualization utilities for training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


def plot_training_curves(
    rewards: List[float],
    losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves for rewards and losses.

    Args:
        rewards: List of episode rewards
        losses: List of training losses
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    # Plot losses
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_success_rate(
    success_rates: List[float],
    window_size: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot success rate over time.

    Args:
        success_rates: List of success rates
        window_size: Window size for moving average
        save_path: Path to save the plot
    """
    # Calculate moving average
    if len(success_rates) >= window_size:
        moving_avg = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
        episodes = np.arange(window_size-1, len(success_rates))
    else:
        moving_avg = success_rates
        episodes = np.arange(len(success_rates))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, moving_avg)
    plt.title('Success Rate Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_action_distribution(
    actions: List[int],
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of actions taken.

    Args:
        actions: List of actions taken
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(actions, bins=range(max(actions)+2), alpha=0.7, edgecolor='black')
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_q_values(
    q_values: np.ndarray,
    actions: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot Q-values for different actions.

    Args:
        q_values: Q-values for each action
        actions: List of action names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(actions, q_values)
    plt.title('Q-Values for Different Actions')
    plt.xlabel('Action')
    plt.ylabel('Q-Value')
    plt.xticks(rotation=45)

    # Color bars based on values
    colors = plt.cm.viridis(q_values / np.max(q_values))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_training_dashboard(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive training dashboard.

    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < len(axes):
            axes[i].plot(values)
            axes[i].set_title(metric_name)
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True)

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
