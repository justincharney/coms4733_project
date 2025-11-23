import os
import sys

import cv2
import gym
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())


def visualize():
    print("Initializing environment...")
    # Initialize environment (headless)
    env = gym.make(
        "gym_grasper:Grasper-v0", image_height=200, image_width=200, render=False
    )
    env.reset()

    # Get observation (RGB image)
    obs = env.get_observation(show=False)
    rgb = obs["rgb"]

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Define the bounds currently set in Grasping_Agent_multidiscrete.py
    min_x, max_x = -0.3, 0.3
    min_y, max_y = -0.749, -0.35
    z = env.TABLE_HEIGHT

    print(f"Visualizing bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")

    # Define corners in world coordinates (Clockwise order)
    corners_world = [
        [min_x, min_y, z],  # Bottom-Left
        [min_x, max_y, z],  # Top-Left
        [max_x, max_y, z],  # Top-Right
        [max_x, min_y, z],  # Bottom-Right
    ]

    # Project world coordinates to pixel coordinates
    pixels = []
    controller = env.unwrapped.controller

    print("Projecting corners...")
    for pt in corners_world:
        try:
            px, py = controller.world_2_pixel(pt, width=200, height=200)
            # Clip to image dimensions for safety
            px = int(np.clip(px, 0, 199))
            py = int(np.clip(py, 0, 199))
            pixels.append((px, py))
            print(f"World {pt} -> Pixel ({px}, {py})")
        except Exception as e:
            print(f"Error projecting point {pt}: {e}")
            return

    # Draw the bounding box
    color = (0, 255, 0)  # Green
    thickness = 2

    for i in range(4):
        p1 = pixels[i]
        p2 = pixels[(i + 1) % 4]
        cv2.line(bgr, p1, p2, color, thickness)

    # Draw corners for clarity
    for p in pixels:
        cv2.circle(bgr, p, 3, (0, 0, 255), -1)  # Red dots at corners

    output_path = "utils/workspace_bounds_viz.png"
    cv2.imwrite(output_path, bgr)
    print(f"Saved visualization to {output_path}")

    env.close()


if __name__ == "__main__":
    visualize()
