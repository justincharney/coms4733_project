#!/usr/bin/env python3
"""
Demo script to visualize a trained SAC model in action.

Loads a trained model and runs it on the grasping environment with rendering,
showing the model's behavior on random episodes.

Usage:
    python demo.py --model-path SAC_Agent/Models/sac_finetune_final.pt --episodes 5
    python demo.py --model-path SAC_Agent/Models/sac_finetune_final.pt --episodes 10
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from SAC_Agent.SAC import SAC
from gym_grasper.envs.GraspingEnv import GraspEnv
from termcolor import colored


def run_demo(
    model_path,
    n_episodes=5,
    render=True,
    max_episode_steps=100,
    delay_between_episodes=2.0,
):
    """
    Run demo of trained model with visualization.

    Args:
        model_path: Path to saved model checkpoint
        n_episodes: Number of demo episodes to run
        render: Whether to render the environment
        max_episode_steps: Maximum steps per episode
        delay_between_episodes: Delay in seconds between episodes
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 80)
    print("SAC Model Demo")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("=" * 80)

    # Create environment with rendering
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
        lr=3e-4,  # Not used during demo
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=1000,  # Small buffer, not used during demo
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

    # Statistics
    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    print(f"\nStarting demo ({n_episodes} episodes)...")
    print("Press Ctrl+C to stop early\n")
    print("-" * 80)

    # Demo loop
    for episode in range(1, n_episodes + 1):
        episode_reward = 0.0
        episode_success = 0.0
        episode_length = 0
        done = False

        # Reset environment
        observation = env.reset()

        # Skip if unreachable goal
        if env.unreachable_goal:
            print(
                colored(
                    f"Episode {episode:2d}: Skipped (unreachable goal)",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            episode_rewards.append(-1.0)
            episode_successes.append(0.0)
            episode_lengths.append(0)
            continue

        print(
            colored(
                f"\nEpisode {episode}/{n_episodes}",
                color="cyan",
                attrs=["bold"],
            )
        )
        print(f"Goal position: {env.desired_goal[:2]}")

        # Episode loop
        step_count = 0
        while not done and step_count < max_episode_steps:
            # Select deterministic action (no exploration)
            action = agent.select_action(observation, deterministic=True)

            # Take step
            next_observation, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            step_count += 1

            if info.get("is_success", 0) or info.get("grasp_success", 0):
                episode_success = 1.0

            observation = next_observation

            # Small delay for visualization to allow viewer to update
            if render:
                time.sleep(0.05)  # Delay to make visualization smoother and allow viewer to update

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        episode_lengths.append(episode_length)

        # Print episode result
        if episode_success:
            status = colored("SUCCESS ", color="green", attrs=["bold"])
        else:
            status = colored("FAILED ", color="red", attrs=["bold"])

        print(
            f"  Result: {status} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Length: {episode_length:3d} steps"
        )

        # Delay between episodes
        if episode < n_episodes:
            time.sleep(delay_between_episodes)

    # Print summary
    print("\n" + "=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print(f"Total Episodes: {n_episodes}")
    print(f"Successful Episodes: {np.sum(episode_successes)}")
    print(f"Success Rate: {np.mean(episode_successes):.1%}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} steps")
    print("=" * 80)

    # Cleanup
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demo a trained SAC model with visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --model-path SAC_Agent/Models/sac_finetune_final.pt --episodes 5
  python demo.py --model-path SAC_Agent/Models/sac_finetune_final.pt --episodes 10
  python demo.py --model-path SAC_Agent/Models/sac_finetune_final.pt --episodes 5 --no-render
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
        default=5,
        help="Number of demo episodes to run (default: 5)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (faster but no visualization)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between episodes (default: 2.0)",
    )

    args = parser.parse_args()

    try:
        run_demo(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            max_episode_steps=args.max_episode_steps,
            delay_between_episodes=args.delay,
        )
        print("\nDemo completed successfully!")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
