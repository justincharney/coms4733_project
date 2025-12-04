#!/usr/bin/env python3
"""
Evaluate a trained SAC model on the grasping environment.

Usage:
    python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 100
    python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 100 --render
    python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 100 --save-results
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import time
import csv
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Get script directory for relative paths
script_dir = Path(__file__).parent

from SAC_Agent.SAC import SAC
from SAC_Agent.metrics import MotionQualityMetrics
from gym_grasper.envs.GraspingEnv import GraspEnv


def evaluate_model(
    model_path,
    n_episodes=100,
    render=False,
    save_results=True,
    results_dir="evaluation_results",
    max_episode_steps=100,
):
    """
    Evaluate a trained SAC model.

    Args:
        model_path: Path to saved model checkpoint
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        save_results: Whether to save results to CSV
        results_dir: Directory to save evaluation results
        max_episode_steps: Maximum steps per episode (to prevent infinite episodes)

    Returns:
        Dictionary with evaluation metrics
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("=" * 80)

    # Setup results directory
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = results_path / f"evaluation_{model_path.stem}_{timestamp}.csv"

    # Create environment
    env = GraspEnv(
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=False,
        render=render,
    )

    # Initialize SAC agent (must match training configuration)
    agent = SAC(
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        goal_dim=3,  # Goal conditioning
        pixel_action_dim=40000,  # 200 * 200
        rotation_action_dim=6,
        lr=3e-4,  # Not used during evaluation
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=1000,  # Small buffer, not used during evaluation
        her_strategy="future",
        her_k=4,
    )

    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        agent.load(str(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Set agent to evaluation mode (deterministic actions)
    agent.actor_network.eval()
    agent.q_network_1.eval()
    agent.q_network_2.eval()

    # Initialize metrics tracker
    metrics_tracker = MotionQualityMetrics(table_height=env.TABLE_HEIGHT)

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

        # Reset environment and metrics
        observation = env.reset()
        metrics_tracker.reset()

        # Skip if unreachable goal
        if env.unreachable_goal:
            episode_rewards.append(-1.0)
            episode_successes.append(0.0)
            episode_lengths.append(0)
            episode_motion_metrics.append({
                "mean_jerk": 0.0,
                "rms_acceleration": 0.0,
                "total_collisions": 0,
            })
            pbar.set_postfix({"Status": "Skipped (unreachable goal)"})
            continue

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
            # Select deterministic action (no exploration)
            action = agent.select_action(observation, deterministic=True)

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

            episode_reward += reward
            episode_length += 1
            step_count += 1

            if info.get("is_success", 0) or info.get("grasp_success", 0):
                episode_success = 1.0

            observation = next_observation

        # Compute motion quality metrics
        motion_metrics = metrics_tracker.compute_all_metrics()

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        episode_lengths.append(episode_length)
        episode_motion_metrics.append({
            "mean_jerk": motion_metrics["mean_jerk"],
            "rms_acceleration": motion_metrics["rms_acceleration"],
            "total_collisions": motion_metrics["total_collisions"],
        })

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
            "Length": episode_length,
        })

    total_time = time.time() - start_time

    # Compute summary statistics
    success_rate = np.mean(episode_successes)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # Motion quality statistics
    mean_jerks = [m["mean_jerk"] for m in episode_motion_metrics]
    rms_accs = [m["rms_acceleration"] for m in episode_motion_metrics]
    total_collisions = [m["total_collisions"] for m in episode_motion_metrics]

    mean_jerk = np.mean(mean_jerks)
    mean_rms_acc = np.mean(rms_accs)
    mean_collisions = np.mean(total_collisions)

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total Episodes: {n_episodes}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Episode: {total_time / n_episodes:.2f} seconds")
    print()
    print("Success Metrics:")
    print(f"  Success Rate: {success_rate:.2%} ({np.sum(episode_successes)}/{n_episodes})")
    print(f"  Successful Episodes: {np.sum(episode_successes)}")
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
    print()
    print("Motion Quality Metrics:")
    print(f"  Mean Jerk: {mean_jerk:.4f}")
    print(f"  Mean RMS Acceleration: {mean_rms_acc:.4f}")
    print(f"  Mean Collisions per Episode: {mean_collisions:.2f}")
    print(f"  Total Collisions: {np.sum(total_collisions)}")
    print("=" * 80)

    # Save results to CSV
    if save_results:
        print(f"\nSaving results to: {csv_file}")
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "reward",
                    "success",
                    "length",
                    "mean_jerk",
                    "rms_acceleration",
                    "collisions",
                ],
            )
            writer.writeheader()
            for i in range(n_episodes):
                writer.writerow({
                    "episode": i + 1,
                    "reward": episode_rewards[i],
                    "success": episode_successes[i],
                    "length": episode_lengths[i],
                    "mean_jerk": episode_motion_metrics[i]["mean_jerk"],
                    "rms_acceleration": episode_motion_metrics[i]["rms_acceleration"],
                    "collisions": episode_motion_metrics[i]["total_collisions"],
                })
        print("Results saved successfully!")

    # Return summary dictionary
    summary = {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "mean_jerk": mean_jerk,
        "mean_rms_acc": mean_rms_acc,
        "mean_collisions": mean_collisions,
        "total_time": total_time,
        "csv_file": str(csv_file) if save_results else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 100
  python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 50 --render
  python evaluate_model.py --model-path Models/sac_finetune_final.pt --episodes 100 --no-save
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
        help="Directory to save evaluation results (default: SAC_Agent/evaluation_results)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)",
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
