"""
Training script for robotic grasping RL agents.
"""

import argparse
import yaml
import torch

from src.environments import GraspingEnvironment
from src.agents import DQNAgent
from src.training import Trainer
from src.utils import set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train robotic grasping RL agent')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    if args.seed is not None:
        config['training']['seed'] = args.seed
    set_seed(config['training']['seed'])

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    config['training']['device'] = str(device)

    print(f"Using device: {device}")
    print(f"Configuration: {config}")

    # Create environment
    env_config = load_config(config['training']['environment']['config_path'])
    env = GraspingEnvironment(env_config)

    # Create agent
    agent_config = load_config(config['training']['agent']['config_path'])
    agent = DQNAgent(
        env.observation_space, env.action_space, agent_config, device
        )

    # Create trainer
    trainer = Trainer(env, agent, config['training'])

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
