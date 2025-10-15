"""
Visualization script for training results and analysis.
"""

import argparse
import yaml
from pathlib import Path

from src.utils.visualization import (
    plot_training_curves,
    plot_success_rate,
    plot_action_distribution,
    create_training_dashboard
)
from src.utils.data_utils import load_episode_data
from src.utils.metrics import compute_comprehensive_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Directory to save plots')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training configuration file')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to analyze (default: all)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.log_dir}")
    print(f"Saving plots to: {args.output_dir}")

    # Load episode data
    episode_files = list(Path(args.log_dir).glob("episode_*.pkl"))
    if not episode_files:
        print("No episode files found!")
        return

    episodes = []
    for file_path in episode_files:
        episode_data = load_episode_data(str(file_path))
        episodes.append(episode_data)

    # Limit episodes if specified
    if args.episodes is not None:
        episodes = episodes[:args.episodes]

    print(f"Loaded {len(episodes)} episodes")

    # Compute metrics
    metrics = compute_comprehensive_metrics(episodes)

    # Extract data for plotting
    rewards = [ep.get('total_reward', 0) for ep in episodes]
    successes = [ep.get('success', False) for ep in episodes]
    lengths = [ep.get('episode_length', 0) for ep in episodes]
    actions = []
    for ep in episodes:
        actions.extend(ep.get('actions', []))

    # Create plots
    print("Creating training curves...")
    plot_training_curves(
        rewards,
        [],  # No loss data available
        save_path=f"{args.output_dir}/training_curves.png"
    )

    print("Creating success rate plot...")
    plot_success_rate(
        successes,
        save_path=f"{args.output_dir}/success_rate.png"
    )

    print("Creating action distribution plot...")
    plot_action_distribution(
        actions,
        save_path=f"{args.output_dir}/action_distribution.png"
    )

    print("Creating comprehensive dashboard...")
    dashboard_metrics = {
        'Rewards': rewards,
        'Success Rate': successes,
        'Episode Length': lengths
    }
    create_training_dashboard(
        dashboard_metrics,
        save_path=f"{args.output_dir}/training_dashboard.png"
    )

    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Total Episodes: {len(episodes)}")
    print(f"Success Rate: {metrics['success_rate']:.3f}")
    print(f"Average Reward: {metrics['average_reward']:.3f}")
    print(f"Average Episode Length: {metrics['average_length']:.1f}")
    print(f"Grasp Efficiency: {metrics['grasp_efficiency']['efficiency']:.3f}")

    if metrics['convergence_metrics']['converged']:
        print(f"Converged at episode: {metrics['convergence_metrics']['convergence_episode']}")
    else:
        print("Training did not converge")

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
