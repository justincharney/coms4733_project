#!/usr/bin/env python3

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

from SAC_Agent.SAC import SAC
from SAC_Agent.training_logger import TrainingLogger
from SAC_Agent.metrics import MotionQualityMetrics
from gym_grasper.envs.GraspingEnv import GraspEnv


def train_finetune_stage(
    n_episodes=1000,
    save_interval=10,
    model_name="sac_finetune",
    log_dir="logs",
    models_dir="Models",
    pretrained_path=None,
    max_episode_steps=100,
):
    """
    Stage 2: Fine-tune SAC model on full GraspEnv with robot motion and HER.

    Args:
        n_episodes: Number of training episodes
        save_interval: Save checkpoint every N episodes if improving
        model_name: Name for the model
        log_dir: Directory for log files
        models_dir: Directory for model checkpoints
        pretrained_path: Path to pretrained model from Stage 1 (optional)
        max_episode_steps: Maximum number of steps per episode (default: 100)
    """
    # Setup directories
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = TrainingLogger(model_name, log_dir=log_dir)
    logger.log("Starting Stage 2: Full SAC Training with HER")
    logger.log(f"Episodes: {n_episodes}, Save Interval: {save_interval}")

    # Create environment
    env = GraspEnv(
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=False,
        render=False,
    )

    # Initialize SAC agent (with goal conditioning for full mode)
    agent = SAC(
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        goal_dim=3,  # Goal conditioning in full mode
        pixel_action_dim=40000,  # 200 * 200
        rotation_action_dim=6,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=100000,
        her_strategy="future",
        her_k=4,
    )

    # Load pretrained weights if available
    # Auto-detect pretrained model if not specified
    if not pretrained_path:
        # Try to find pretrained model in default location
        default_pretrained = models_path / "pretrain_vision_best.pt"
        if default_pretrained.exists():
            pretrained_path = str(default_pretrained)
            logger.log(f"No pretrained path specified. Auto-detected: {pretrained_path}")
        else:
            logger.log("No pretrained model path specified and no default model found.", level="WARNING")
            logger.log(f"Expected location: {default_pretrained}", level="WARNING")
            logger.log("Starting training from scratch")
    
    if pretrained_path and Path(pretrained_path).exists():
        logger.log(f"Loading pretrained model from: {pretrained_path}")
        try:
            # Try strict loading first
            try:
                agent.load(pretrained_path, strict=True)
                logger.log("Pretrained model loaded successfully (strict mode)")
            except RuntimeError as e:
                # If strict loading fails, try partial loading
                logger.log(f"Strict loading failed, attempting partial loading: {e}", level="WARNING")
                agent.load(pretrained_path, strict=False)
                logger.log("Pretrained model loaded successfully (partial mode - encoder and embeddings transferred)")
        except Exception as e:
            logger.log(f"Warning: Could not load pretrained model: {e}", level="WARNING")
            logger.log("Starting training from scratch")
    elif pretrained_path:
        logger.log(f"Pretrained model not found at: {pretrained_path}", level="WARNING")
        logger.log("Starting training from scratch")

    logger.log("Agent initialized")

    # Initialize metrics tracker
    metrics_tracker = MotionQualityMetrics(table_height=env.TABLE_HEIGHT)

    # Training statistics
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    best_mean_reward = float("-inf")

    start_time = time.time()

    # Training loop with progress bar
    pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="episode")
    for episode in pbar:
        episode_reward = 0.0
        episode_success = 0.0
        episode_length = 0
        done = False

        # Reset environment and metrics
        observation = env.reset()

        # Abort episode immediately if no reachable goal
        if env.unreachable_goal:
            episode_reward = -1.0
            episode_success = 0.0
            episode_length = 0
            done = True

            # Store transition for logging
            agent.store_transition(
                observation, (0, 0), episode_reward, observation, done,
                {"is_success": 0.0, "unreachable_goal": True}
            )
            agent.end_episode()

            # Record statistics
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            episode_lengths.append(episode_length)

            # Skip rest of episode
            continue

        agent.episode_buffer = []  # Clear episode buffer
        metrics_tracker.reset()

        # Get initial state for metrics
        try:
            metrics_tracker.update(
                mujoco_data=env.data,
                actuated_joint_ids=env.controller.actuated_joint_ids,
                ee_body_id=env.ee_body_id,
                dt=env.dt,
            )
        except Exception:
            pass  # Metrics update may fail on first step

        # Episode loop
        step_count = 0
        while not done and step_count < max_episode_steps:
            # Select action
            action = agent.select_action(observation, deterministic=False)
            step_count += 1

            # Take step
            next_observation, reward, done, info = env.step(action)

            # Update metrics
            try:
                metrics_tracker.update(
                    mujoco_data=env.data,
                    actuated_joint_ids=env.controller.actuated_joint_ids,
                    ee_body_id=env.ee_body_id,
                    dt=env.dt,
                )
            except Exception:
                pass  # Metrics update may fail occasionally

            # Store transition
            agent.store_transition(
                observation, action, reward, next_observation, done, info
            )

            episode_reward += reward
            episode_length += 1
            if info.get("is_success", 0) or info.get("grasp_success", 0):
                episode_success = 1.0

            observation = next_observation

            # Update agent
            if len(agent.replay_buffer) > 256:  # Minimum buffer size
                loss_info = agent.update(batch_size=256)
            else:
                loss_info = None

        # Check if episode was truncated due to step limit
        if step_count >= max_episode_steps and not done:
            # Store final transition with done=True to mark truncation
            agent.store_transition(
                observation, (0, 0), 0.0, observation, True,
                {"is_success": 0.0, "truncated": True}
            )

        # End episode (triggers HER)
        agent.end_episode()

        # Compute motion quality metrics
        motion_metrics = metrics_tracker.compute_all_metrics()

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        episode_lengths.append(episode_length)

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
            "Length": episode_length,
        })

        # Log episode
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            success_rate=success_rate,
            loss_info=loss_info,
            metrics={
                "length": episode_length,
                "mean_jerk": motion_metrics["mean_jerk"],
                "rms_acc": motion_metrics["rms_acceleration"],
                "collisions": motion_metrics["total_collisions"],
            },
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

    parser = argparse.ArgumentParser(description="Stage 2: Fine-tune SAC model")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save interval"
    )
    parser.add_argument("--model-name", type=str, default="sac_finetune", help="Model name")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory (default: SAC_Agent/logs)")
    parser.add_argument("--models-dir", type=str, default=None, help="Models directory (default: SAC_Agent/Models)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model from Stage 1",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode (default: 100)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    log_dir = args.log_dir if args.log_dir else str(script_dir / "logs")
    models_dir = args.models_dir if args.models_dir else str(script_dir / "Models")

    train_finetune_stage(
        n_episodes=args.episodes,
        save_interval=args.save_interval,
        model_name=args.model_name,
        log_dir=log_dir,
        models_dir=models_dir,
        pretrained_path=args.pretrained,
        max_episode_steps=args.max_episode_steps,
    )
