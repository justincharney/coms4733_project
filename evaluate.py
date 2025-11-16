#!/usr/bin/env python3
"""
Evaluation script for trained grasping models.

This script loads trained models (DQN or SAC), runs evaluation episodes,
and reports detailed metrics including task success rate, collision rate,
trajectory smoothness, and control quality.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import numpy as np
from termcolor import colored

# IMPORTANT: Set this BEFORE importing mujoco_py or gym
os.environ["MUJOCO_GL"] = "osmesa"

sys.path.insert(0, str(Path(__file__).parent))

from Grasping_Agent_multidiscrete import Grasp_Agent
from SAC_Agent.SAC_agent import SAC_Agent
from metrics import (
    MetricsEvaluator,
    extract_contact_info,
    get_cube_position,
    get_joint_states,
)


def detect_agent_type(model_path):
    """
    Detect the agent type from the checkpoint file.

    Args:
        model_path: Path to the model checkpoint file

    Returns:
        str: Either 'SAC' or 'DQN'
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        # SAC checkpoints have 'actor_state_dict' and 'critic1_state_dict'
        # DQN checkpoints have 'model_state_dict'
        if 'actor_state_dict' in checkpoint or 'critic1_state_dict' in checkpoint:
            return 'SAC'
        elif 'model_state_dict' in checkpoint:
            return 'DQN'
    except Exception:
        pass

    # Fallback to filename detection
    model_path_str = str(model_path).upper()
    if 'SAC' in model_path_str:
        return 'SAC'
    elif 'DQN' in model_path_str:
        return 'DQN'

    # Default to DQN if uncertain
    return 'DQN'


def load_agent(model_path, agent_type, seed, depth_only=False, render=False):
    """
    Load the appropriate agent based on type.

    Args:
        model_path: Path to the model checkpoint
        agent_type: 'SAC' or 'DQN'
        seed: Random seed
        depth_only: Whether to use depth-only observations
        render: Whether to enable visualization (default: False for faster evaluation)

    Returns:
        Loaded agent instance
    """
    # Monkey-patch gym.make to use render=False when not rendering
    # This prevents the viewer from being created in the first place
    original_make = None
    if not render:
        import gym
        original_make = gym.make

        def patched_make(*args, **kwargs):
            # Force render=False if it's our grasping environment
            if 'gym_grasper' in str(args[0]) or 'Grasper' in str(args[0]):
                kwargs['render'] = False
            return original_make(*args, **kwargs)

        gym.make = patched_make

    try:
        if agent_type == 'SAC':
            agent = SAC_Agent(
                seed=seed,
                load_path=model_path,
                train=False,
                depth_only=depth_only,
            )
        else:  # DQN
            agent = Grasp_Agent(
                seed=seed,
                load_path=model_path,
                train=False,
                depth_only=depth_only,
            )

        # Ensure rendering is disabled
        if not render:
            env = agent.env.unwrapped
            env.render = False
            # Disable viewer to prevent visualization
            if hasattr(env.controller, 'viewer'):
                try:
                    # Close and remove viewer if it exists
                    if env.controller.viewer is not None:
                        # Try to close the viewer properly
                        if hasattr(env.controller.viewer, 'close'):
                            try:
                                env.controller.viewer.close()
                            except Exception:
                                pass
                        # Try to delete the viewer
                        if hasattr(env.controller.viewer, '__del__'):
                            try:
                                del env.controller.viewer
                            except Exception:
                                pass
                    # Set viewer to False to prevent rendering
                    env.controller.viewer = False
                except Exception:
                    # If closing fails, just set to False
                    env.controller.viewer = False
    finally:
        # Restore original gym.make
        if original_make is not None:
            import gym
            gym.make = original_make

    return agent


def get_simulation_from_env(env):
    """
    Extract MuJoCo simulation object from environment.

    Args:
        env: Gym environment (GraspEnv)

    Returns:
        MuJoCo simulation object
    """
    # Access the unwrapped environment
    unwrapped = env.unwrapped
    # Get simulation from controller
    return unwrapped.controller.sim


def evaluate_model(
    model_path,
    n_episodes=10,
    steps_per_episode=10,
    seed=999,
    depth_only=False,
    fast_inference=False,
    verbose=True,
    render=False,
):
    """
    Evaluate a trained model and compute comprehensive metrics.

    Args:
        model_path: Path to trained model checkpoint
        n_episodes: Number of evaluation episodes
        steps_per_episode: Maximum steps per episode
        seed: Random seed for reproducibility
        depth_only: Whether to use depth-only observations
        fast_inference: For SAC, skip critic network for faster inference
        verbose: Whether to print detailed progress

    Returns:
        Dictionary containing aggregate metrics
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(colored("\n" + "=" * 70, color="cyan", attrs=["bold"]))
        print(colored("MODEL EVALUATION", color="cyan", attrs=["bold"]))
        print(colored("=" * 70, color="cyan", attrs=["bold"]))
        print(colored(f"Model: {model_path}", color="green"))
        print(colored(f"Episodes: {n_episodes}", color="yellow"))
        print(colored(f"Steps per episode: {steps_per_episode}", color="yellow"))
        print(colored(f"Seed: {seed}", color="yellow"))

    # Detect agent type
    agent_type = detect_agent_type(model_path)
    if verbose:
        print(colored(f"Agent type: {agent_type}", color="yellow"))

    # Load agent
    if verbose:
        print(colored("Loading model...", color="blue"))
        if not render:
            print(colored("Rendering disabled for faster evaluation", color="yellow"))
    agent = load_agent(
        model_path, agent_type, seed, depth_only=depth_only, render=render
    )

    if verbose:
        print(colored("Model loaded successfully!", color="green"))

    # Initialize metrics evaluator
    # Bin target position from GraspingEnv.move_and_grasp (line 318)
    bin_target_pos = np.array([0.6, 0.0, 1.15])
    bin_region_size = np.array([0.2, 0.3, 0.1])  # Approximate bin size
    position_tolerance = 0.05  # 5cm tolerance for settling time

    evaluator = MetricsEvaluator(
        bin_target_pos=bin_target_pos,
        bin_region_size=bin_region_size,
        position_tolerance=position_tolerance,
        dt=0.002,  # MuJoCo default timestep
    )

    # Get simulation object for metrics
    sim = get_simulation_from_env(agent.env)
    controller = agent.env.unwrapped.controller

    # Track basic statistics
    total_rewards = 0
    successful_grasps = 0

    if verbose:
        print(colored("\nStarting evaluation...", color="blue"))

    # Run evaluation episodes
    for episode in range(1, n_episodes + 1):
        if verbose:
            print(colored(f"\n{'=' * 70}", color="cyan"))
            print(colored(f"EPISODE {episode}/{n_episodes}", color="cyan", attrs=["bold"]))
            print(colored(f"{'=' * 70}", color="cyan"))

        # Start episode tracking
        evaluator.start_episode()

        # Reset environment
        state = agent.env.reset()
        state = agent.transform_observation(
            state, normalize=True, jitter_and_noise=False
        )

        episode_reward = 0
        episode_done = False

        # Run episode steps
        for step in range(1, steps_per_episode + 1):
            if episode_done:
                break

            # Get action from model (greedy policy)
            if agent_type == 'SAC' and fast_inference:
                action_idx, q_value = agent.greedy(state, return_q_value=False)
            else:
                action_idx, q_value = agent.greedy(state)
            env_action = agent.transform_action(action_idx)

            # Collect metrics data before step
            cube_pos = get_cube_position(sim, cube_body_name="pick_box")
            joint_pos, joint_vel = get_joint_states(
                sim, joint_ids=controller.actuated_joint_ids
            )
            contacts = extract_contact_info(sim)

            # Execute action (render parameter is controlled by env.render flag)
            next_state, reward, done, _ = agent.env.unwrapped.step(
                env_action,
                record_grasps=False,
                action_info=agent.last_action,
            )

            # Add step data to metrics
            evaluator.add_step(
                cube_pos=cube_pos,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                contacts=contacts,
                reward=reward,
            )

            # Update statistics
            episode_reward += reward
            total_rewards += reward
            if reward > 0:
                successful_grasps += 1

            if verbose and step % 5 == 0:
                pixel_x = env_action[0] % agent.env.IMAGE_WIDTH
                pixel_y = env_action[0] // agent.env.IMAGE_WIDTH
                rotation = env_action[1]
                msg = (
                    f"  Step {step}: Action=({pixel_x},{pixel_y},rot={rotation}), "
                    f"Reward={reward}, Q={q_value:.3f}"
                )
                print(colored(msg, color="white"))

            # Transform next state
            state = agent.transform_observation(
                next_state, normalize=True, jitter_and_noise=False
            )

            if done:
                episode_done = True
                if verbose:
                    print(colored(f"  Episode finished early at step {step}", color="yellow"))

        # End episode and compute metrics
        episode_metrics = evaluator.end_episode()

        if verbose:
            print(colored(f"\nEpisode {episode} Summary:", color="magenta", attrs=["bold"]))
            print(f"  Total Reward: {episode_reward}")
            print(f"  Task Success: {episode_metrics['task_success']}")
            print(f"  Final Position Error: {episode_metrics['final_position_error']:.4f} m")
            print(f"  Collision Rate: {episode_metrics['collision_rate']:.4f}")
            print(f"  Mean Jerk: {episode_metrics['mean_jerk']:.4f}")
            print(f"  RMS Joint Acceleration: {episode_metrics['rms_joint_acceleration']:.4f}")
            print(f"  Overshoot: {episode_metrics['overshoot']:.4f} m")
            if episode_metrics['settling_time'] != float('inf'):
                print(f"  Settling Time: {episode_metrics['settling_time']:.4f} s")
            else:
                print("  Settling Time: Never settled")

    aggregate_metrics = evaluator.get_aggregate_metrics()

    # Add basic statistics
    aggregate_metrics['total_rewards'] = total_rewards
    aggregate_metrics['successful_grasps'] = successful_grasps
    aggregate_metrics['total_steps'] = n_episodes * steps_per_episode
    total_steps = n_episodes * steps_per_episode
    if total_steps > 0:
        grasp_rate = successful_grasps / total_steps
    else:
        grasp_rate = 0.0
    aggregate_metrics['grasp_success_rate'] = grasp_rate
    if n_episodes > 0:
        avg_reward = total_rewards / n_episodes
    else:
        avg_reward = 0.0
    aggregate_metrics['avg_reward_per_episode'] = avg_reward

    if verbose:
        print(colored("\n" + "=" * 70, color="green", attrs=["bold"]))
        print(colored("EVALUATION SUMMARY", color="green", attrs=["bold"]))
        print(colored("=" * 70, color="green", attrs=["bold"]))
        evaluator.print_summary()
        print(f"Total Rewards: {total_rewards}")
        print(f"Successful Grasps: {successful_grasps}")
        print(f"Grasp Success Rate: {aggregate_metrics['grasp_success_rate']:.2%}")
        avg_reward = aggregate_metrics['avg_reward_per_episode']
        print(f"Average Reward per Episode: {avg_reward:.2f}")
        print(colored("=" * 70 + "\n", color="green", attrs=["bold"]))

    agent.env.close()

    return aggregate_metrics


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained grasping models with comprehensive metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Maximum steps per episode (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Random seed (default: 999)",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="Use depth-only observations",
    )
    parser.add_argument(
        "--fast-inference",
        action="store_true",
        help="For SAC: skip critic network for faster inference",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable visualization (slower, default: disabled for speed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save metrics to JSON file",
    )

    args = parser.parse_args()

    try:
        # Run evaluation
        metrics = evaluate_model(
            model_path=args.model,
            n_episodes=args.episodes,
            steps_per_episode=args.steps,
            seed=args.seed,
            depth_only=args.depth_only,
            fast_inference=args.fast_inference,
            verbose=not args.quiet,
            render=args.render,
        )

        # Save to file if requested
        if args.output:
            import json
            # Convert numpy types to native Python types for JSON
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_json[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                elif value == float('inf'):
                    metrics_json[key] = None
                else:
                    metrics_json[key] = value

            with open(args.output, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(colored(f"\nMetrics saved to: {args.output}", color="green"))

        return metrics

    except FileNotFoundError as e:
        print(colored(f"\nERROR: {e}", color="red", attrs=["bold"]))
        # List available models
        models_dir = Path("Models")
        if models_dir.exists():
            print(f"\nAvailable model files in {models_dir}:")
            for f in sorted(models_dir.glob("*_weights.pt")):
                print(f"  - {f}")
        sys.exit(1)
    except Exception as e:
        print(colored(f"\nERROR: {e}", color="red", attrs=["bold"]))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
