#!/usr/bin/env python3

"""
Example agent script with offscreen rendering.
This avoids creating an interactive MjViewer window while still capturing images.
Works completely headless without X11/xvfb.

The key is:
- render=False: Don't create interactive viewer (no GLFW/X11 needed)
- MUJOCO_GL=osmesa: Use software rendering for sim.render() image capture
- Observations still work via sim.render() in get_image_data()
"""

import os

# IMPORTANT: Set this BEFORE importing mujoco_py or gym
os.environ["MUJOCO_GL"] = "osmesa"

import gym
import numpy as np
from termcolor import colored
import time
import cv2 as cv

print(colored("\n=== Offscreen Rendering Mode ===", color="cyan", attrs=["bold"]))
print("This mode works completely headless without X11/xvfb")
print("Images are captured via sim.render() using OSMesa\n")

# render=False means no interactive viewer window is created
# Observations still work because get_image_data() uses sim.render()
env = gym.make("gym_grasper:Grasper-v0", show_obs=False, render=False)

N_EPISODES = 3
N_STEPS = 10
SAVE_OBSERVATIONS = True  # Set to True to save observation images

env.print_info()

print(
    colored(f"\nRunning {N_EPISODES} episodes with {N_STEPS} steps each", color="green")
)
print(f"Saving observations: {SAVE_OBSERVATIONS}\n")

# Create directory for saved images
if SAVE_OBSERVATIONS:
    os.makedirs("observations", exist_ok=True)

total_steps = 0
total_rewards = 0

for episode in range(1, N_EPISODES + 1):
    print(colored(f"\n{'=' * 60}", color="cyan"))
    print(colored(f"EPISODE {episode}/{N_EPISODES}", color="cyan", attrs=["bold"]))
    print(colored(f"{'=' * 60}\n", color="cyan"))

    obs = env.reset()
    episode_reward = 0

    # Save initial observation
    if SAVE_OBSERVATIONS and "rgb" in obs:
        rgb = obs["rgb"]
        filename = f"observations/episode_{episode}_step_0_init.png"
        cv.imwrite(filename, cv.cvtColor(rgb.astype(np.uint8), cv.COLOR_RGB2BGR))
        print(f"Saved initial observation: {filename}")

    for step in range(N_STEPS):
        print(colored(f"\n--- Step {step + 1}/{N_STEPS} ---", color="yellow"))

        # Sample random action
        action = env.action_space.sample()

        try:
            # Use unwrapped to access custom step method
            # record_grasps=False since we're not using an interactive viewer
            observation, reward, done, info = env.unwrapped.step(
                action, record_grasps=False, markers=False
            )

            episode_reward += reward
            total_steps += 1

            # Save observation image if enabled
            if SAVE_OBSERVATIONS and "rgb" in observation:
                rgb = observation["rgb"]
                filename = f"observations/episode_{episode}_step_{step + 1}_reward_{reward}.png"
                cv.imwrite(
                    filename, cv.cvtColor(rgb.astype(np.uint8), cv.COLOR_RGB2BGR)
                )
                print(f"Saved observation: {filename}")

            if reward > 0:
                print(
                    colored(
                        f"✓ SUCCESS! Reward: {reward}", color="green", attrs=["bold"]
                    )
                )
            else:
                print(f"Reward: {reward}")

            # Print observation shape for debugging
            if "rgb" in observation and "depth" in observation:
                print(
                    f"Observation shapes - RGB: {observation['rgb'].shape}, Depth: {observation['depth'].shape}"
                )

            if done:
                print(
                    colored(
                        f"Episode finished early at step {step + 1}", color="yellow"
                    )
                )
                break

        except Exception as e:
            print(colored(f"✗ Error during step: {e}", color="red", attrs=["bold"]))
            import traceback

            traceback.print_exc()
            break

    print(
        colored(
            f"\nEpisode {episode} Total Reward: {episode_reward}",
            color="magenta",
            attrs=["bold"],
        )
    )
    total_rewards += episode_reward

env.close()

print(colored("\n" + "=" * 60, color="green", attrs=["bold"]))
print(
    colored("=== Offscreen Rendering Test Complete ===", color="green", attrs=["bold"])
)
print(colored("=" * 60 + "\n", color="green", attrs=["bold"]))
print(f"Total steps executed: {total_steps}")
print(f"Total rewards collected: {total_rewards}")
print(f"Average reward per episode: {total_rewards / N_EPISODES:.2f}")

if SAVE_OBSERVATIONS:
    print(f"\nObservation images saved to: observations/")
    print("You can download these to view the robot's perspective")

print(
    colored("\n✓ Offscreen mode completed successfully!", color="green", attrs=["bold"])
)
print("This proves the environment works without any display/X11!\n")
