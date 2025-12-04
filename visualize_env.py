#!/usr/bin/env python3
"""
Enhanced script to visualize the grasping simulation environment.
Displays:
- RGB and depth observations from multiple camera views
- Workspace boundaries
- Reachable workspace visualization
"""

import os
import sys

import cv2
import gym
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())


def draw_workspace_bounds(image, env, color=(0, 255, 0), thickness=2):
    """
    Draw workspace boundaries on the image.

    Args:
        image: BGR image to draw on
        env: Environment instance
        color: BGR color tuple
        thickness: Line thickness
    """
    workspace_bounds = env.unwrapped.workspace_bounds
    x_min, x_max = workspace_bounds["x"]
    y_min, y_max = workspace_bounds["y"]
    z = env.unwrapped.TABLE_HEIGHT

    # Define corners in world coordinates
    corners_world = [
        [x_min, y_min, z],  # Bottom-Left
        [x_min, y_max, z],  # Top-Left
        [x_max, y_max, z],  # Top-Right
        [x_max, y_min, z],  # Bottom-Right
    ]

    controller = env.unwrapped.controller
    pixels = []

    for pt in corners_world:
        try:
            px, py = controller.world_2_pixel(pt, width=200, height=200)
            px = int(np.clip(px, 0, 199))
            py = int(np.clip(py, 0, 199))
            pixels.append((px, py))
        except Exception:
            return  # Skip if projection fails

    # Draw bounding box
    if len(pixels) == 4:
        for i in range(4):
            p1 = pixels[i]
            p2 = pixels[(i + 1) % 4]
            cv2.line(image, p1, p2, color, thickness)

        # Draw corners
        for p in pixels:
            cv2.circle(image, p, 3, (0, 0, 255), -1)  # Red dots at corners


def visualize_reachable_workspace(image, env, grid_size=20, point_size=1):
    """
    Visualize reachable workspace by sampling points and checking IK.

    Args:
        image: BGR image to draw on
        env: Environment instance
        grid_size: Number of points per dimension to sample
        point_size: Size of points to draw
    """
    workspace_bounds = env.unwrapped.workspace_bounds
    x_min, x_max = workspace_bounds["x"]
    y_min, y_max = workspace_bounds["y"]
    z = env.unwrapped.TABLE_HEIGHT

    x_range = np.linspace(x_min, x_max, grid_size)
    y_range = np.linspace(y_min, y_max, grid_size)

    controller = env.unwrapped.controller
    reachable_count = 0

    for y in y_range:
        for x in x_range:
            target_pos = np.array([x, y, z])

            # Check Inverse Kinematics
            joint_angles = controller.ik(target_pos)
            is_reachable = joint_angles is not None

            # Project to pixel coordinates
            try:
                px, py = controller.world_2_pixel(
                    target_pos, width=200, height=200
                )
                px, py = int(px), int(py)

                # Draw on image if within bounds
                if 0 <= px < 200 and 0 <= py < 200:
                    if is_reachable:
                        cv2.circle(image, (px, py), point_size, (0, 255, 0), -1)  # Green
                        reachable_count += 1
                    else:
                        cv2.circle(image, (px, py), point_size, (0, 0, 255), -1)  # Red
            except Exception:
                pass

    return reachable_count


def get_camera_view(env, camera_name, width=200, height=200):
    """
    Get RGB and depth images from a specific camera.

    Args:
        env: Environment instance
        camera_name: Name of the camera (e.g., "top_down", "side", "main1")
        width: Image width
        height: Image height

    Returns:
        rgb, depth: RGB and depth images
    """
    rgb, depth = env.unwrapped.controller.get_image_data(
        camera=camera_name, width=width, height=height, show=False
    )
    depth = env.unwrapped.controller.depth_2_meters(depth)
    return rgb, depth


def create_info_panel(text_lines, width=400, height=300):
    """
    Create a text information panel.

    Args:
        text_lines: List of strings to display
        width: Panel width
        height: Panel height

    Returns:
        BGR image with text
    """
    panel = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)  # Black text
    thickness = 1
    line_height = 20
    y_offset = 20

    for i, line in enumerate(text_lines):
        y = y_offset + i * line_height
        cv2.putText(panel, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return panel


def visualize_environment(
    render_mode="opencv",
    num_steps=0,
    show_workspace=True,
    show_reachable=True,
    show_all_cameras=True,
    include_main1=False,
):
    """
    Visualize the grasping environment with comprehensive information.

    Args:
        render_mode: "opencv" to show in OpenCV window, "save" to save image
        num_steps: Number of random steps to take (0 = just show initial state)
        show_workspace: Whether to show workspace boundaries
        show_reachable: Whether to show reachable workspace
        show_all_cameras: Whether to show views from all cameras
        include_main1: Whether to include main1 camera (often produces blank images)
    """
    print("Initializing environment...")

    # Initialize environment
    env = gym.make(
        "gym_grasper:Grasper-v0",
        image_height=200,
        image_width=200,
        show_obs=False,
        render=False,
    )

    print("Resetting environment...")
    obs = env.reset()

    print(f"Observation shape - RGB: {obs['rgb'].shape}, Depth: {obs['depth'].shape}")
    print(f"Desired goal: {obs['desired_goal']}")
    print(f"Achieved goal: {obs['achieved_goal']}")
    print(f"Workspace bounds: {env.unwrapped.workspace_bounds}")

    # Get views from different cameras
    # Note: main1 camera is positioned far away (pos="2 2 2.7") and often shows
    # blank/blue images because it's looking at empty space/background.
    # It's excluded by default, but can be included with include_main1=True
    cameras = ["top_down", "side"] if show_all_cameras else ["top_down"]
    if include_main1:
        cameras.append("main1")
    camera_views = {}

    print("\nCapturing views from cameras...")
    for camera_name in cameras:
        try:
            rgb, depth = get_camera_view(env, camera_name)
            # Check if image is mostly empty/blue (likely a bad camera view)
            # If more than 80% of pixels are very dark or very uniform, skip it
            rgb_mean = np.mean(rgb)
            rgb_std = np.std(rgb)
            if rgb_std < 5 or rgb_mean < 10:  # Very uniform or very dark
                print(f"  ⚠ {camera_name} camera produced blank/uniform image (skipping)")
                continue

            camera_views[camera_name] = {
                "rgb": rgb,
                "depth": depth
            }
            print(f"  ✓ {camera_name} camera")
        except Exception as e:
            print(f"  ✗ {camera_name} camera failed: {e}")

    # Process top-down view (main view)
    if "top_down" in camera_views:
        rgb = camera_views["top_down"]["rgb"]
        depth = camera_views["top_down"]["depth"]

        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Create a copy for annotations
        annotated = bgr.copy()

        # Draw workspace boundaries
        if show_workspace:
            print("\nDrawing workspace boundaries...")
            draw_workspace_bounds(annotated, env, color=(0, 255, 0), thickness=2)

        # Visualize reachable workspace
        if show_reachable:
            print("Calculating reachable workspace (this may take a moment)...")
            reachable_count = visualize_reachable_workspace(
                annotated, env, grid_size=15, point_size=1
            )
            print(f"  Found {reachable_count} reachable points")

        # Normalize depth for visualization
        depth_normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Add text labels
        cv2.putText(
            annotated, "Top-Down View (Annotated)", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        cv2.putText(
            depth_colored, "Depth Map", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )

        top_down_row = np.hstack([annotated, depth_colored])
    else:
        top_down_row = None

    # Create multi-camera view
    if show_all_cameras and len(camera_views) > 1:
        camera_rows = []
        for camera_name in cameras:
            if camera_name in camera_views:
                rgb = camera_views[camera_name]["rgb"]
                depth = camera_views[camera_name]["depth"]

                bgr_cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                depth_norm = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                depth_col = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                # Add labels
                cv2.putText(
                    bgr_cam, f"{camera_name} - RGB", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
                cv2.putText(
                    depth_col, f"{camera_name} - Depth", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )

                camera_rows.append(np.hstack([bgr_cam, depth_col]))

        if camera_rows:
            multi_camera_view = np.vstack(camera_rows)
        else:
            multi_camera_view = None
    else:
        multi_camera_view = None

    # Create information panel
    info_lines = [
        "Environment Information:",
        "",
        "Workspace Bounds:",
        f"  X: [{env.unwrapped.workspace_bounds['x'][0]:.3f}, {env.unwrapped.workspace_bounds['x'][1]:.3f}]",
        f"  Y: [{env.unwrapped.workspace_bounds['y'][0]:.3f}, {env.unwrapped.workspace_bounds['y'][1]:.3f}]",
        f"  Z (Table Height): {env.unwrapped.TABLE_HEIGHT:.3f}",
        "",
        f"Desired Goal: {obs['desired_goal']}",
        f"Achieved Goal: {obs['achieved_goal']}",
        "",
        "Legend:",
        "  Green box = Workspace bounds",
        "  Green dots = Reachable points",
        "  Red dots = Unreachable points",
    ]

    info_panel = create_info_panel(info_lines, width=400, height=300)

    # Combine all visualizations
    if top_down_row is not None:
        if multi_camera_view is not None:
            # Combine top-down with multi-camera view
            left_side = np.vstack([top_down_row, multi_camera_view])
            # Resize info panel to match height
            info_height = left_side.shape[0]
            info_panel_resized = cv2.resize(info_panel, (400, info_height))
            combined = np.hstack([left_side, info_panel_resized])
        else:
            # Just top-down with info panel
            info_panel_resized = cv2.resize(info_panel, (400, top_down_row.shape[0]))
            combined = np.hstack([top_down_row, info_panel_resized])
    else:
        combined = info_panel

    if render_mode == "opencv":
        # Display in OpenCV window
        window_name = "Grasping Environment - Comprehensive View"
        cv2.imshow(window_name, combined)

        print("\n" + "=" * 60)
        print("Visualization window opened!")
        print("Click on the OpenCV window and press any key (or 'q') to close")
        print("Or press Ctrl+C in the terminal to force close")
        print("=" * 60)

        _ = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    elif render_mode == "save":
        # Save images
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)

        # Save main combined view
        combined_path = os.path.join(output_dir, "env_comprehensive.png")
        cv2.imwrite(combined_path, combined)
        print(f"\nSaved comprehensive visualization to: {combined_path}")

        # Save individual camera views
        for camera_name, views in camera_views.items():
            rgb = views["rgb"]
            depth = views["depth"]
            bgr_cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_norm = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_col = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            rgb_path = os.path.join(output_dir, f"env_{camera_name}_rgb.png")
            depth_path = os.path.join(output_dir, f"env_{camera_name}_depth.png")
            cv2.imwrite(rgb_path, bgr_cam)
            cv2.imwrite(depth_path, depth_col)
            print(f"  - {camera_name} RGB: {rgb_path}")
            print(f"  - {camera_name} Depth: {depth_path}")

        # Save annotated top-down view if available
        if "top_down" in camera_views and (show_workspace or show_reachable):
            annotated_path = os.path.join(output_dir, "env_top_down_annotated.png")
            cv2.imwrite(annotated_path, annotated)
            print(f"  - Annotated top-down: {annotated_path}")

    # Optionally take some random steps
    if num_steps > 0:
        print(f"\nTaking {num_steps} random steps...")
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            print(f"Step {step + 1}: Reward = {reward:.3f}, Done = {done}")
            if done:
                print("Episode finished, resetting...")
                env.reset()
                break

    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize the grasping simulation environment with comprehensive information"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="opencv",
        choices=["opencv", "save"],
        help="Visualization mode: 'opencv' to display window, 'save' to save images",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of random steps to take (0 = just show initial state)",
    )
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Disable workspace boundaries visualization",
    )
    parser.add_argument(
        "--no-reachable",
        action="store_true",
        help="Disable reachable workspace visualization",
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help="Only show top-down camera view (disable multi-camera view)",
    )
    parser.add_argument(
        "--include-main1",
        action="store_true",
        help="Include main1 camera (note: often produces blank/blue images)",
    )

    args = parser.parse_args()

    visualize_environment(
        render_mode=args.mode,
        num_steps=args.steps,
        show_workspace=not args.no_workspace,
        show_reachable=not args.no_reachable,
        show_all_cameras=not args.single_camera,
        include_main1=args.include_main1,
    )
