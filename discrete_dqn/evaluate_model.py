#!/usr/bin/env python3
"""
Evaluate a trained DQN model on the grasping environment.

Usage:
    python evaluate_model.py --model-path Models/dqn_baseline_final.pt --episodes 100
    python evaluate_model.py --model-path Models/dqn_baseline_final.pt --episodes 100 --render
"""

import sys
import argparse
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
from eval_utils import save_evaluation_results, print_evaluation_summary, compute_evaluation_summary
from gym_grasper.envs.FastGraspEnv import FastGraspEnv

# Import motion metrics tracker
sac_agent_path = project_root / "SAC_Agent"
sys.path.insert(0, str(sac_agent_path))
from metrics import MotionQualityMetrics  # type: ignore


def evaluate_model(
    model_path,
    n_episodes=100,
    render=False,
    save_results=True,
    results_dir="evaluation_results",
    use_full_env=False,
    max_episode_steps=100,
):
    """
    Evaluate a trained DQN model.

    Args:
        model_path: Path to saved model checkpoint
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        save_results: Whether to save results to CSV
        results_dir: Directory to save evaluation results
        use_full_env: Whether to use GraspingEnv instead of FastGraspEnv
        max_episode_steps: Maximum steps per episode (to prevent infinite episodes)

    Returns:
        Dictionary with evaluation metrics
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 80)
    print("DQN Model Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("=" * 80)

    # Setup results directory
    csv_file = None
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = results_path / f"evaluation_{model_path.stem}_{timestamp}.csv"

    # Create environment
    # Note: FastGraspEnv doesn't move the robot, so motion metrics will be zero.
    # Use --use-full-env flag to use GraspingEnv for meaningful motion metrics.
    if use_full_env:
        from gym_grasper.envs.GraspingEnv import GraspEnv
        env = GraspEnv(
            image_width=200,
            image_height=200,
            show_obs=False,
            render=render,
        )
        print("Using GraspingEnv (full robot simulation) for evaluation")
    else:
        env = FastGraspEnv(
            image_width=200,
            image_height=200,
            show_obs=False,
            render=render,
        )
        print("Using FastGraspEnv (no robot motion) for evaluation")
        print("  Note: Motion metrics (jerk, acceleration) will be zero with FastGraspEnv.")
        print("  Use --use-full-env flag to get meaningful motion metrics.")

    # Initialize DQN agent (must match training configuration)
    agent = DQNAgent(
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        pixel_action_dim=40000,  # 200 * 200
        rotation_action_dim=6,
        lr=0.0005,  # Not used during evaluation
        buffer_size=2000,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay_steps=8000,
    )

    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        agent.load(str(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Set agent to evaluation mode (deterministic actions, epsilon=0)
    agent.q_network.eval()
    agent.epsilon = 0.0  # No exploration during evaluation

    # Initialize motion quality metrics tracker
    metrics_tracker = MotionQualityMetrics(table_height=0.91)

    # Evaluation statistics
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    episode_motion_metrics = []

    start_time = time.time()

    print(f"\nStarting evaluation ({n_episodes} episodes)...")
    print("-" * 80)

    # Evaluation loop with progress bar
    pbar = tqdm(range(1, n_episodes + 1), desc="Evaluating", unit="episode")
    for _ in pbar:
        episode_reward = 0.0
        episode_success = 0.0
        episode_length = 0
        done = False

        # Reset environment and metrics tracker
        observation = env.reset()
        metrics_tracker.reset()

        # Get action mask for valid pixels (on table)
        action_mask = None
        try:
            if hasattr(env, 'get_reachable_pixels'):
                action_mask = env.get_reachable_pixels()
        except Exception:
            pass  # Action mask not critical

        # Try to update metrics with initial state (FastGraspEnv may not have full MuJoCo state)
        metrics_updated = False
        try:
            if hasattr(env, 'data') and hasattr(env, 'controller') and hasattr(env, 'dt'):
                if hasattr(env.controller, 'actuated_joint_ids') and hasattr(env, 'ee_body_id'):
                    metrics_tracker.update(
                        mujoco_data=env.data,
                        actuated_joint_ids=env.controller.actuated_joint_ids,
                        ee_body_id=env.ee_body_id,
                        dt=env.dt,
                    )
                    metrics_updated = True
        except Exception as e:
            # Metrics update may fail if FastGraspEnv doesn't have full state
            if len(episode_rewards) == 0:  # Only print once
                print(f"Warning: Could not update metrics at reset: {e}")

        # Episode loop
        # FastGraspEnv: done=True after one action (contextual bandit)
        # GraspingEnv: done=True when object is grasped or max steps reached
        step_count = 0
        while not done and step_count < max_episode_steps:
            # Select deterministic action (no exploration)
            action = agent.select_action(observation, deterministic=True, action_mask=action_mask)

            # Take step
            next_observation, reward, done, info = env.step(action)

            # Try to update metrics after step
            try:
                if hasattr(env, 'data') and hasattr(env, 'controller') and hasattr(env, 'dt'):
                    if hasattr(env.controller, 'actuated_joint_ids') and hasattr(env, 'ee_body_id'):
                        metrics_tracker.update(
                            mujoco_data=env.data,
                            actuated_joint_ids=env.controller.actuated_joint_ids,
                            ee_body_id=env.ee_body_id,
                            dt=env.dt,
                        )
                        metrics_updated = True
            except Exception as e:
                # Metrics update may fail if FastGraspEnv doesn't have full state
                if len(episode_rewards) == 0:  # Only print once
                    print(f"Warning: Could not update metrics after step: {e}")

            episode_reward += reward
            episode_length += 1
            step_count += 1
            if info.get("is_success", 0) or reward > 0:
                episode_success = 1.0

            observation = next_observation

        # Compute motion quality metrics for this episode
        try:
            motion_metrics = metrics_tracker.compute_all_metrics()

            # Debug: Check if we have data for motion metrics
            if len(episode_rewards) == 0:  # Only print once
                num_acc_samples = len(metrics_tracker.joint_accelerations)
                num_vel_samples = len(metrics_tracker.joint_velocities)
                num_pos_samples = len(metrics_tracker.joint_positions)
                print(f"Debug: Metrics tracker has {num_pos_samples} position, "
                      f"{num_vel_samples} velocity, {num_acc_samples} acceleration samples")
                print(f"Debug: Metrics updated successfully: {metrics_updated}")
                if isinstance(env, FastGraspEnv):
                    print("Warning: FastGraspEnv does not move the robot, "
                          "so motion metrics (jerk, acceleration) will be zero.")
                    print("         Use GraspingEnv for evaluation if you need motion quality metrics.")

            episode_motion_metrics.append({
                "mean_jerk": motion_metrics.get("mean_jerk", 0.0),
                "rms_acceleration": motion_metrics.get("rms_acceleration", 0.0),
                "total_collisions": motion_metrics.get("total_collisions", 0),
            })
        except Exception as e:
            # If metrics computation fails, use default values
            if len(episode_rewards) == 0:  # Only print once
                print(f"Warning: Could not compute motion metrics: {e}")
            episode_motion_metrics.append({
                "mean_jerk": 0.0,
                "rms_acceleration": 0.0,
                "total_collisions": 0,
            })

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        episode_lengths.append(episode_length)

        # Update progress bar
        if len(episode_successes) >= 10:
            recent_success = np.mean(episode_successes[-10:])
        else:
            recent_success = np.mean(episode_successes) if episode_successes else 0.0
        if len(episode_rewards) >= 10:
            recent_reward = np.mean(episode_rewards[-10:])
        else:
            recent_reward = np.mean(episode_rewards) if episode_rewards else 0.0

        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Mean": f"{recent_reward:.2f}",
            "Success": f"{recent_success:.1%}",
        })

    total_time = time.time() - start_time

    # Print summary using shared utility
    print_evaluation_summary(
        n_episodes=n_episodes,
        episode_rewards=episode_rewards,
        episode_successes=episode_successes,
        episode_lengths=episode_lengths,
        episode_motion_metrics=episode_motion_metrics,
        total_time=total_time,
    )

    # Save results to CSV
    if save_results and csv_file:
        print(f"\nSaving results to: {csv_file}")
        save_evaluation_results(
            csv_file=csv_file,
            episode_rewards=episode_rewards,
            episode_successes=episode_successes,
            episode_lengths=episode_lengths,
            episode_motion_metrics=episode_motion_metrics,
        )
        print("Results saved successfully!")

    # Return summary dictionary
    summary = compute_evaluation_summary(
        n_episodes=n_episodes,
        episode_rewards=episode_rewards,
        episode_successes=episode_successes,
        episode_lengths=episode_lengths,
        episode_motion_metrics=episode_motion_metrics,
        total_time=total_time,
        csv_file=csv_file,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --model-path Models/dqn_baseline_final.pt --episodes 100
  python evaluate_model.py --model-path Models/dqn_baseline_final.pt --episodes 50 --render
  python evaluate_model.py --model-path Models/dqn_baseline_final.pt --episodes 100 --no-save
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to CSV",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: discrete_dqn/evaluation_results)",
    )
    parser.add_argument(
        "--use-full-env",
        action="store_true",
        help="Use GraspingEnv instead of FastGraspEnv for evaluation. "
             "Required for meaningful motion quality metrics (jerk, acceleration). "
             "Note: This will be slower as it actually moves the robot.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100). "
             "Note: FastGraspEnv always terminates after 1 step, so this mainly applies when using --use-full-env.",
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        # Try relative to script directory first, then current working directory
        script_model_path = script_dir / args.model_path
        if script_model_path.exists():
            model_path = script_model_path
        else:
            model_path = Path(args.model_path).resolve()

    results_dir = args.results_dir if args.results_dir else str(script_dir / "evaluation_results")

    try:
        summary = evaluate_model(
            model_path=str(model_path),
            n_episodes=args.episodes,
            render=args.render,
            save_results=not args.no_save,
            results_dir=results_dir,
            use_full_env=args.use_full_env,
            max_episode_steps=args.max_episode_steps,
        )
        print("\nEvaluation completed successfully!")
        if summary["csv_file"]:
            print(f"Results saved to: {summary['csv_file']}")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
