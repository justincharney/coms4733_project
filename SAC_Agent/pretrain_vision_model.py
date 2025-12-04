#!/usr/bin/env python3

import sys
import warnings
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

# Suppress MuJoCo renderer cleanup warnings in headless environments
# Note: The "Exception ignored in: <function Renderer.__del__>" message is harmless.
# It occurs when the MuJoCo renderer is garbage collected but was never properly
# initialized (common in headless/server environments without OpenGL/DISPLAY).
# The training functionality is unaffected - the code already handles renderer
# unavailability by returning blank frames. This is a known MuJoCo Python binding issue.
warnings.filterwarnings('ignore', message='.*Renderer.*', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Get script directory for relative paths
script_dir = Path(__file__).parent

from SAC_Agent.SAC import SAC
from SAC_Agent.training_logger import TrainingLogger
from gym_grasper.envs.FastGraspEnv import FastGraspEnv


def train_pretrain_stage(
    n_episodes=3000,
    save_interval=10,
    model_name="pretrain_vision",
    log_dir="logs",
    models_dir="Models",
):
    """
    Stage 1: Pretrain vision model on FastGraspEnv (no robot motion).

    Args:
        n_episodes: Number of training episodes
        save_interval: Save checkpoint every N episodes if improving
        model_name: Name for the model
        log_dir: Directory for log files
        models_dir: Directory for model checkpoints
    """
    # Setup directories
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(model_name, log_dir=log_dir)
    logger.log("Starting Stage 1: Vision Pretraining")
    logger.log(f"Episodes: {n_episodes}, Save Interval: {save_interval}")

    # Create environment
    env = FastGraspEnv(
        image_width=200,
        image_height=200,
        show_obs=False,
        render=False,
    )

    # Initialize SAC agent (no goal conditioning for fast mode)
    agent = SAC(
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        goal_dim=0,  # No goal conditioning in fast mode
        pixel_action_dim=40000,  # 200 * 200
        rotation_action_dim=6,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=50000,  # Smaller buffer for fast training
        her_strategy="future",
        her_k=4,
    )

    logger.log("Agent initialized")

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

        # Abort episode immediately if no reachable goal
        if env.unreachable_goal:
            episode_reward = -1.0
            episode_success = 0.0
            done = True

            # Store transition for logging
            state = {"rgb": observation["rgb"], "depth": observation["depth"]}
            state["desired_goal"] = np.zeros(3, dtype=np.float32)
            state["achieved_goal"] = np.zeros(3, dtype=np.float32)
            agent.store_transition(
                state, (0, 0), episode_reward, state, done,
                {"is_success": 0.0, "unreachable_goal": True}
            )
            agent.end_episode()

            # Record statistics
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)

            # Skip rest of episode
            continue

        agent.episode_buffer = []  # Clear episode buffer

        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(observation, deterministic=False)

            # Take step
            next_observation, reward, done, info = env.step(action)

            # Store transition (no goal info for fast mode)
            # Create simplified state without goals
            state = {"rgb": observation["rgb"], "depth": observation["depth"]}
            next_state = {
                "rgb": next_observation["rgb"],
                "depth": next_observation["depth"],
            }
            # Add dummy goals for compatibility (will be ignored)
            state["desired_goal"] = np.zeros(3, dtype=np.float32)
            state["achieved_goal"] = np.zeros(3, dtype=np.float32)
            next_state["desired_goal"] = np.zeros(3, dtype=np.float32)
            next_state["achieved_goal"] = np.zeros(3, dtype=np.float32)

            agent.store_transition(state, action, reward, next_state, done, info)

            episode_reward += reward
            if info.get("is_success", 0):
                episode_success = 1.0

            observation = next_observation

            # Update agent
            if len(agent.replay_buffer) > 256:  # Minimum buffer size
                loss_info = agent.update(batch_size=256)
            else:
                loss_info = None

        # End episode (no HER in fast mode, but we still call it for consistency)
        agent.end_episode()

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)

        # Compute running statistics
        recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
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
        })

        # Log episode
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            success_rate=success_rate,
            loss_info=loss_info,
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

    parser = argparse.ArgumentParser(description="Stage 1: Pretrain vision model")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--save-interval", type=int, default=10, help="Save interval")
    parser.add_argument("--model-name", type=str, default="pretrain_vision", help="Model name")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory (default: SAC_Agent/logs)")
    parser.add_argument("--models-dir", type=str, default=None, help="Models directory (default: SAC_Agent/Models)")

    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    log_dir = args.log_dir if args.log_dir else str(script_dir / "logs")
    models_dir = args.models_dir if args.models_dir else str(script_dir / "Models")

    train_pretrain_stage(
        n_episodes=args.episodes,
        save_interval=args.save_interval,
        model_name=args.model_name,
        log_dir=log_dir,
        models_dir=models_dir,
    )
