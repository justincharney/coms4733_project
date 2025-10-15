"""
Evaluation script for trained robotic grasping RL agents.
"""

import argparse
import yaml
import torch

from src.environments import GraspingEnvironment
from src.agents import DQNAgent
from src.training import Evaluator
from src.utils import set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained robotic grasping RL agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes during evaluation')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save videos of evaluation episodes')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config['training']['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Evaluating {args.num_episodes} episodes")

    # Create environment
    env_config = load_config(config['training']['environment']['config_path'])
    env = GraspingEnvironment(env_config)

    # Create agent
    agent_config = load_config(config['training']['agent']['config_path'])
    agent = DQNAgent(env.observation_space, env.action_space, agent_config, device)

    # Load trained model
    agent.load_checkpoint(args.checkpoint)

    # Create evaluator
    evaluator = Evaluator(env, agent, args.output_dir)

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render,
        save_videos=args.save_videos
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {results['success_rate']:.3f}")
    print(f"Average Reward: {results['average_reward']:.3f}")
    print(f"Average Episode Length: {results['average_length']:.1f}")
    print(f"Grasp Efficiency: {results['grasp_efficiency']:.3f}")


if __name__ == '__main__':
    main()
