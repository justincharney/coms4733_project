#!/usr/bin/env python3

"""
DQN Training Script

Trains the DQN base network on the grasping environment.
Based on the milestone document specification with:
- Epsilon-greedy exploration (1.0 to 0.2 over 8000 steps)
- Binary cross-entropy loss
- Replay buffer (2000 entries)
- No target network (gamma=0, immediate reward prediction)
"""

import sys
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Get script directory for relative paths
script_dir = Path(__file__).parent

from dqn_agent import DQNAgent
from training_logger import TrainingLogger
from gym_grasper.envs.FastGraspEnv import FastGraspEnv


def train_dqn(
    n_episodes=1480,
    save_interval=10,
    model_name="dqn_baseline",
    log_dir="logs",
    models_dir="Models",
    batch_size=32,
):
    """
    Train DQN agent on FastGraspEnv.

    Args:
        n_episodes: Number of training episodes (default: 1480 as per milestone)
        save_interval: Save checkpoint every N episodes if improving
        model_name: Name for the model
        log_dir: Directory for log files
        models_dir: Directory for model checkpoints
        batch_size: Batch size for training
    """
    # Setup directories
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(model_name, log_dir=log_dir)
    logger.log("Starting DQN Training")
    logger.log(f"Episodes: {n_episodes}, Save Interval: {save_interval}")
    logger.log(f"Batch Size: {batch_size}, Buffer Size: 2000")

    # Create environment (FastGraspEnv for pixel-based rewards)
    env = FastGraspEnv(
        image_width=200,
        image_height=200,
        show_obs=False,
        render=False,
    )

    # Initialize DQN agent
    agent = DQNAgent(
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        pixel_action_dim=40000,  # 200 * 200
        rotation_action_dim=6,
        lr=0.0005,  # As per milestone document
        buffer_size=2000,  # As per milestone document
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=8000,
    )

    logger.log("Agent initialized")
    logger.log(f"Initial epsilon: {agent.epsilon:.3f}")

    # Training statistics
    episode_rewards = []
    episode_successes = []
    best_mean_reward = float("-inf")

    start_time = time.time()

    # Training loop with progress bar
    pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="episode")
    for episode in pbar:
        episode_reward = 0.0
        episode_success = 0.0
        done = False

        # Reset environment
        observation = env.reset()

        # Get action mask for valid pixels (on table)
        action_mask = None
        try:
            # Try to get reachable pixels mask if available
            if hasattr(env, 'get_reachable_pixels'):
                action_mask = env.get_reachable_pixels()
        except Exception:
            pass  # Action mask not critical

        # Select action
        action = agent.select_action(observation, deterministic=False, action_mask=action_mask)

        # Take step (FastGraspEnv: done=True after one action)
        next_observation, reward, done, info = env.step(action)

        # Store transition
        agent.store_transition(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

        episode_reward = reward
        if info.get("is_success", 0) or reward > 0:
            episode_success = 1.0

        # Update agent (learn from replay buffer)
        loss_info = agent.learn(batch_size=batch_size)

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)

        # Compute running statistics
        recent_rewards = (
            episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        )
        mean_reward = np.mean(recent_rewards)
        success_rate = (
            np.mean(episode_successes[-100:])
            if len(episode_successes) >= 100
            else np.mean(episode_successes)
        )

        # Update progress bar
        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Mean": f"{mean_reward:.2f}",
            "Success": f"{success_rate:.1%}",
            "Epsilon": f"{agent.epsilon:.3f}",
        })

        # Log episode
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            success_rate=success_rate,
            loss_info=loss_info,
            epsilon=agent.epsilon,
        )

        # Checkpoint saving (every save_interval episodes if improving)
        if episode % save_interval == 0:
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

                # Save checkpoint
                checkpoint_path = models_path / f"{model_name}_best.pt"
                agent.save(str(checkpoint_path))
                logger.log_checkpoint(episode, mean_reward, str(checkpoint_path))

    # Save final model
    final_path = models_path / f"{model_name}_final.pt"
    agent.save(str(final_path))
    logger.log(f"Final model saved to: {final_path}")

    # Training summary
    total_time = time.time() - start_time
    best_reward = max(episode_rewards) if episode_rewards else 0.0
    final_reward = episode_rewards[-1] if episode_rewards else 0.0
    logger.log_summary(n_episodes, best_reward, final_reward, total_time)

    logger.close()

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN baseline")
    parser.add_argument("--episodes", type=int, default=600, help="Number of episodes")
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save interval"
    )
    parser.add_argument("--model-name", type=str, default="dqn_baseline", help="Model name")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory (default: discrete_dqn/logs)")
    parser.add_argument("--models-dir", type=str, default=None, help="Models directory (default: discrete_dqn/Models)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    log_dir = args.log_dir if args.log_dir else str(script_dir / "logs")
    models_dir = args.models_dir if args.models_dir else str(script_dir / "Models")

    train_dqn(
        n_episodes=args.episodes,
        save_interval=args.save_interval,
        model_name=args.model_name,
        log_dir=log_dir,
        models_dir=models_dir,
        batch_size=args.batch_size,
    )
