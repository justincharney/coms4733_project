"""
Shared utilities for evaluation and plotting.
Common functions that can be reused across different agent types.
"""

import numpy as np
import csv
from pathlib import Path


def save_evaluation_results(
    csv_file,
    episode_rewards,
    episode_successes,
    episode_lengths,
    episode_motion_metrics=None,
):
    """
    Save evaluation results to CSV file.

    Args:
        csv_file: Path to CSV file
        episode_rewards: List of episode rewards
        episode_successes: List of episode success flags
        episode_lengths: List of episode lengths
        episode_motion_metrics: Optional list of motion metrics dicts
    """
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["episode", "reward", "success", "length"]
    if episode_motion_metrics:
        fieldnames.extend(["mean_jerk", "rms_acceleration", "collisions"])

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(episode_rewards)):
            row = {
                "episode": i + 1,
                "reward": episode_rewards[i],
                "success": episode_successes[i],
                "length": episode_lengths[i],
            }
            if episode_motion_metrics:
                row.update({
                    "mean_jerk": episode_motion_metrics[i].get("mean_jerk", 0.0),
                    "rms_acceleration": episode_motion_metrics[i].get("rms_acceleration", 0.0),
                    "collisions": episode_motion_metrics[i].get("total_collisions", 0),
                })
            writer.writerow(row)


def print_evaluation_summary(
    n_episodes,
    episode_rewards,
    episode_successes,
    episode_lengths,
    episode_motion_metrics=None,
    total_time=None,
):
    """
    Print evaluation summary statistics.

    Args:
        n_episodes: Total number of episodes
        episode_rewards: List of episode rewards
        episode_successes: List of episode success flags
        episode_lengths: List of episode lengths
        episode_motion_metrics: Optional list of motion metrics dicts
        total_time: Total evaluation time in seconds
    """
    # Compute summary statistics
    success_rate = np.mean(episode_successes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total Episodes: {n_episodes}")
    if total_time is not None:
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time per Episode: {total_time / n_episodes:.2f} seconds")
    print()
    print("Success Metrics:")
    print(f"  Success Rate: {success_rate:.2%} ({np.sum(episode_successes):.1f}/{n_episodes})")
    print(f"  Successful Episodes: {np.sum(episode_successes):.1f}")
    print()
    print("Reward Metrics:")
    print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Best Reward: {np.max(episode_rewards):.3f}")
    print(f"  Worst Reward: {np.min(episode_rewards):.3f}")
    print(f"  Median Reward: {np.median(episode_rewards):.3f}")
    print()
    print("Episode Length Metrics:")
    print(f"  Mean Length: {mean_length:.1f} ± {std_length:.1f} steps")
    print(f"  Shortest Episode: {np.min(episode_lengths)} steps")
    print(f"  Longest Episode: {np.max(episode_lengths)} steps")

    # Motion quality statistics (if available)
    if episode_motion_metrics:
        mean_jerks = [m.get("mean_jerk", 0.0) for m in episode_motion_metrics]
        rms_accs = [m.get("rms_acceleration", 0.0) for m in episode_motion_metrics]
        total_collisions = [m.get("total_collisions", 0) for m in episode_motion_metrics]

        mean_jerk = np.mean(mean_jerks)
        mean_rms_acc = np.mean(rms_accs)
        mean_collisions = np.mean(total_collisions)

        print()
        print("Motion Quality Metrics:")
        print(f"  Mean Jerk: {mean_jerk:.4f}")
        print(f"  Mean RMS Acceleration: {mean_rms_acc:.4f}")
        print(f"  Mean Collisions per Episode: {mean_collisions:.2f}")
        print(f"  Total Collisions: {np.sum(total_collisions)}")

    print("=" * 80)


def compute_evaluation_summary(
    n_episodes,
    episode_rewards,
    episode_successes,
    episode_lengths,
    episode_motion_metrics=None,
    total_time=None,
    csv_file=None,
):
    """
    Compute and return evaluation summary dictionary.

    Args:
        n_episodes: Total number of episodes
        episode_rewards: List of episode rewards
        episode_successes: List of episode success flags
        episode_lengths: List of episode lengths
        episode_motion_metrics: Optional list of motion metrics dicts
        total_time: Total evaluation time in seconds
        csv_file: Path to CSV file if saved

    Returns:
        Dictionary with summary statistics
    """
    success_rate = np.mean(episode_successes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    summary = {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "total_time": total_time,
        "csv_file": str(csv_file) if csv_file else None,
    }

    if episode_motion_metrics:
        mean_jerks = [m.get("mean_jerk", 0.0) for m in episode_motion_metrics]
        rms_accs = [m.get("rms_acceleration", 0.0) for m in episode_motion_metrics]
        total_collisions = [m.get("total_collisions", 0) for m in episode_motion_metrics]

        summary.update({
            "mean_jerk": np.mean(mean_jerks),
            "mean_rms_acc": np.mean(rms_accs),
            "mean_collisions": np.mean(total_collisions),
        })

    return summary
