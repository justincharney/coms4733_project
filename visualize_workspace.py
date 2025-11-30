#!/usr/bin/env python3
"""
Visualize the workspace by moving the arm to different positions and capturing images.
"""

import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if os.uname().sysname != "Darwin" else "glfw"

import cv2 as cv
import gym
import numpy as np
from termcolor import colored


def capture_arm_at_position(env, position, name, cameras, output_dir):
    """Move arm to a position and capture images from multiple cameras."""
    print(colored(f"\nMoving arm to {name}: {position}", "cyan"))

    result = env.controller.move_ee(
        position,
        max_steps=2000,
        quiet=True,
        render=False,
        tolerance=0.05,
    )
    print(f"  Move result: {result}")
    env.controller.stay(200, render=False)  # Let arm settle

    for camera in cameras:
        rgb, _ = env.controller.get_image_data(
            width=800, height=600, camera=camera, show=False
        )
        output_path = output_dir / f"arm_at_{name}_{camera}.png"
        cv.imwrite(str(output_path), cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
        print(colored(f"  Saved {output_path}", "green"))


def main():
    # Create environment
    env = gym.make(
        "gym_grasper:Grasper-v0",
        image_height=200,
        image_width=200,
        show_obs=False,
        render=False,
    )

    # Reset to get objects on table
    env.reset()
    env.controller.stay(500, render=False)  # Let objects settle

    output_dir = Path("renders")
    output_dir.mkdir(exist_ok=True)

    cameras = ["top_down_wide", "side"]

    # Positions to visualize the arm at
    arm_positions = {
        "drop_current": np.array([0.6, 0.0, 0.92]),
    }

    # Move arm to each position and capture
    for name, pos in arm_positions.items():
        capture_arm_at_position(env, pos, name, cameras, output_dir)

    print(colored("\n" + "=" * 60, "yellow"))
    print(colored("Arm position visualization complete!", "yellow", attrs=["bold"]))
    print(colored("=" * 60, "yellow"))

    env.close()


if __name__ == "__main__":
    main()
