#!/usr/bin/env python3

"""
Test script for evaluating a trained grasping model with visual simulation.
This script loads a trained model checkpoint and runs it with greedy actions,
showing the visual simulation of the robot performing grasps.

Supports both DQN (Grasp_Agent) and SAC (SAC_Agent) models.
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to path to import Grasping_Agent
sys.path.insert(0, str(Path(__file__).parent))

from Grasping_Agent_multidiscrete import Grasp_Agent  # noqa: E402
from termcolor import colored  # noqa: E402
from SAC_Agent.SAC_agent import SAC_Agent

# Configuration
# Update this to your trained model path
# Examples:
# MODEL_PATH = (
#     "Models/DQN_RESNET_LR_0.0005_OPTIM_ADAM_H_200_W_200_STEPS_100_"
#     "BUFFER_SIZE_2000_BATCH_SIZE_12_SEED_999_11_12_2025_1_2_weights.pt"
# )
# MODEL_PATH = (
#     "Models/SAC_SAC_RESNET_LR_0.0003_OPTIM_ADAM_H_200_W_200_STEPS_4_"
#     "BUFFER_SIZE_2000_BATCH_SIZE_12_SEED_999_11_13_2025_3_6_weights.pt"
# )

MODEL_PATH = (
    "Models/SAC_SAC_RESNET_LR_0.0003_OPTIM_ADAM_H_200_W_200_STEPS_4_"
    "BUFFER_SIZE_2000_BATCH_SIZE_12_SEED_999_11_13_2025_3_6_weights.pt"
)

N_EPISODES = 5
STEPS_PER_EPISODE = 10
SEED = 999
# Set to False for faster inference (skips critic network, ~2x faster for SAC)
# Note: Q-values will be approximate (from actor logits) instead of true Q-values
FAST_INFERENCE = False


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


def load_agent(model_path, agent_type, seed, depth_only=False):
    """
    Load the appropriate agent based on type.

    Args:
        model_path: Path to the model checkpoint
        agent_type: 'SAC' or 'DQN'
        seed: Random seed
        depth_only: Whether to use depth-only observations

    Returns:
        Loaded agent instance
    """
    if agent_type == 'SAC':
        agent = SAC_Agent(
            seed=seed,
            load_path=model_path,
            train=False,  # This enables visual rendering
            depth_only=depth_only,
        )
    else:  # DQN
        agent = Grasp_Agent(
            seed=seed,
            load_path=model_path,
            train=False,  # This enables visual rendering
            depth_only=depth_only,
        )
    return agent


def main():
    print(colored("\n" + "=" * 60, color="cyan", attrs=["bold"]))
    print(colored("Testing Trained Grasping Model", color="cyan", attrs=["bold"]))
    print(colored("=" * 60 + "\n", color="cyan", attrs=["bold"]))

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(
            colored(
                f"ERROR: Model file not found: {MODEL_PATH}",
                color="red",
                attrs=["bold"],
            )
        )
        print("\nPlease update MODEL_PATH in the script to point to your "
              "trained model.")
        # Check for Models directory
        models_dir = Path("Models")
        if models_dir.exists():
            print(f"\nAvailable model files in {models_dir}:")
            for f in sorted(models_dir.glob("*_weights.pt")):
                print(f"  - {f}")
        else:
            print("Example model files in current directory:")
            for f in os.listdir("."):
                if f.endswith("_weights.pt"):
                    print(f"  - {f}")
        return

    print(colored(f"Loading model from: {MODEL_PATH}", color="green"))

    # Detect agent type
    agent_type = detect_agent_type(MODEL_PATH)
    print(colored(f"Detected agent type: {agent_type}", color="yellow"))

    # Load the appropriate agent
    agent = load_agent(MODEL_PATH, agent_type, SEED, depth_only=False)

    print(colored("Model loaded successfully!", color="green"))
    print(
        colored(
            f"Running {N_EPISODES} episodes with {STEPS_PER_EPISODE} "
            f"steps each\n",
            color="yellow",
        )
    )

    total_rewards = 0
    successful_grasps = 0

    for episode in range(1, N_EPISODES + 1):
        print(colored("\n" + "=" * 60, color="cyan"))
        print(colored(f"EPISODE {episode}/{N_EPISODES}", color="cyan",
                      attrs=["bold"]))
        print(colored("=" * 60 + "\n", color="cyan"))

        # Reset environment
        state = agent.env.reset()
        state = agent.transform_observation(
            state, normalize=True, jitter_and_noise=False
        )

        episode_reward = 0

        for step in range(1, STEPS_PER_EPISODE + 1):
            print(colored(f"--- Step {step}/{STEPS_PER_EPISODE} ---",
                          color="yellow"))

            # Get greedy action from trained model
            # For SAC: FAST_INFERENCE=True skips critic network (2x faster)
            # For DQN: FAST_INFERENCE has no effect (only one network)
            if agent_type == 'SAC' and FAST_INFERENCE:
                action_idx, q_value = agent.greedy(state, return_q_value=False)
            else:
                action_idx, q_value = agent.greedy(state)
            env_action = agent.transform_action(action_idx)

            pixel_x = env_action[0] % agent.env.IMAGE_WIDTH
            pixel_y = env_action[0] // agent.env.IMAGE_WIDTH
            rotation = env_action[1]
            print(
                f"Action: Pixel X={pixel_x}, Pixel Y={pixel_y}, "
                f"Rotation={rotation}, Q-value={q_value:.4f}"
            )

            # Execute action in environment
            next_state, reward, done, _ = agent.env.unwrapped.step(
                env_action,
                record_grasps=True,
                action_info=agent.last_action,
            )

            episode_reward += reward
            total_rewards += reward

            if reward > 0:
                successful_grasps += 1
                print(
                    colored(
                        f"✓ SUCCESS! Grasped object. Reward: {reward}",
                        color="green",
                        attrs=["bold"],
                    )
                )
            else:
                print(f"Reward: {reward}")

            # Transform next state for next iteration
            state = agent.transform_observation(
                next_state, normalize=True, jitter_and_noise=False
            )

            if done:
                print(
                    colored(
                        f"Episode finished early at step {step}",
                        color="yellow",
                    )
                )
                break

        print(
            colored(
                f"\nEpisode {episode} Total Reward: {episode_reward}",
                color="magenta",
                attrs=["bold"],
            )
        )

    # Print summary
    print(colored("\n" + "=" * 60, color="green", attrs=["bold"]))
    print(colored("Test Summary", color="green", attrs=["bold"]))
    print(colored("=" * 60, color="green", attrs=["bold"]))
    print(f"Total Episodes: {N_EPISODES}")
    print(f"Total Steps: {N_EPISODES * STEPS_PER_EPISODE}")
    print(f"Successful Grasps: {successful_grasps}")
    total_steps = N_EPISODES * STEPS_PER_EPISODE
    success_rate = successful_grasps / total_steps * 100
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Rewards: {total_rewards}")
    avg_reward = total_rewards / N_EPISODES
    print(f"Average Reward per Episode: {avg_reward:.2f}")
    print(colored("=" * 60 + "\n", color="green", attrs=["bold"]))

    agent.env.close()
    print(colored("Test completed!", color="green", attrs=["bold"]))


if __name__ == "__main__":
    main()
