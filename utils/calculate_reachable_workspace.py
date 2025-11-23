import os
import sys

import cv2
import gym
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())


def calculate_reachability():
    print("Initializing environment...")
    env = gym.make(
        "gym_grasper:Grasper-v0", image_height=200, image_width=200, render=False
    )
    env.reset()

    # Get the background image
    obs = env.get_observation(show=False)
    rgb = obs["rgb"]
    bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Define a broad search grid covering the entire potential table area
    # Robot base is at (0,0). Table is roughly y in [-1.0, -0.3], x in [-0.5, 0.5]
    x_range = np.linspace(-0.5, 0.5, 40)
    y_range = np.linspace(-1.0, -0.3, 40)
    z = env.TABLE_HEIGHT

    reachable_points = []

    print(f"Scanning grid ({len(x_range)}x{len(y_range)} points)...")

    for y in y_range:
        for x in x_range:
            target_pos = np.array([x, y, z])

            # Check Inverse Kinematics
            joint_angles = env.unwrapped.controller.ik(target_pos)
            is_reachable = joint_angles is not None

            # Project to pixel coordinates for visualization
            try:
                px, py = env.unwrapped.controller.world_2_pixel(
                    target_pos, width=200, height=200
                )
                px, py = int(px), int(py)

                # Draw on image if within bounds
                if 0 <= px < 200 and 0 <= py < 200:
                    color = (0, 255, 0) if is_reachable else (0, 0, 255)  # Green vs Red
                    cv2.circle(bgr_image, (px, py), 2, color, -1)
            except Exception:
                pass

            if is_reachable:
                reachable_points.append((x, y))

    # Calculate statistics
    if reachable_points:
        reachable_points = np.array(reachable_points)
        min_x, min_y = np.min(reachable_points, axis=0)
        max_x, max_y = np.max(reachable_points, axis=0)

        print("\n--- Reachable Workspace Statistics ---")
        print(f"Total reachable points found: {len(reachable_points)}")
        print(f"Suggested Bounds (Raw):")
        print(f"  X: [{min_x:.3f}, {max_x:.3f}]")
        print(f"  Y: [{min_y:.3f}, {max_y:.3f}]")

        # Draw the suggested bounding box in Blue
        try:
            p1 = env.unwrapped.controller.world_2_pixel(
                [min_x, min_y, z], width=200, height=200
            )
            p2 = env.unwrapped.controller.world_2_pixel(
                [min_x, max_y, z], width=200, height=200
            )
            p3 = env.unwrapped.controller.world_2_pixel(
                [max_x, max_y, z], width=200, height=200
            )
            p4 = env.unwrapped.controller.world_2_pixel(
                [max_x, min_y, z], width=200, height=200
            )

            pts = np.array([p1, p2, p3, p4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(bgr_image, [pts], True, (255, 0, 0), 2)  # Blue box
        except Exception:
            pass

    else:
        print("No reachable points found!")

    output_path = "utils/reachable_workspace.png"
    cv2.imwrite(output_path, bgr_image)
    print(f"\nVisualization saved to {output_path}")

    env.close()


if __name__ == "__main__":
    calculate_reachability()
