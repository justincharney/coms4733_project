#!/usr/bin/env python3
"""
Test script to verify pixel to world coordinate mapping.
Checks if different pixel coordinates map to different world coordinates.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from gym_grasper.envs.GraspingEnv import GraspEnv


def test_pixel_to_world_mapping():
    """Test pixel to world coordinate mapping with random pixels."""

    print("=" * 60)
    print("Testing Pixel to World Coordinate Mapping")
    print("=" * 60)

    # Create environment
    env = GraspEnv(
        image_width=200,
        image_height=200,
        show_obs=False,
        demo=False,
        render=False,
    )

    # Reset environment to get initial observation
    print("\nResetting environment...")
    obs = env.reset()

    # Get depth image
    depth = obs["depth"]
    print(f"Depth image shape: {depth.shape}")
    print(f"Depth value range: [{depth.min():.4f}, {depth.max():.4f}]")
    valid_pixels = np.sum((depth > 0) & (depth <= 2.0))
    print(f"Valid depth pixels (>0 and <=2.0): {valid_pixels} / {depth.size}")

    print("\n" + "-" * 60)
    print("Testing 10 random pixel coordinates:")
    print("-" * 60)

    mappings = []
    valid_count = 0

    rng = np.random.default_rng(seed=42)
    for i in range(10):
        # Random pixel coordinates
        px = rng.integers(0, env.IMAGE_WIDTH)
        py = rng.integers(0, env.IMAGE_HEIGHT)

        # Get depth at this pixel
        depth_value = depth[py, px]

        # Check if depth is valid
        is_valid = depth_value > 0 and depth_value <= 2.0

        if is_valid:
            # Convert pixel to world coordinates
            coords = env.controller.pixel_2_world(
                pixel_x=px,
                pixel_y=py,
                depth=depth_value,
                height=env.IMAGE_HEIGHT,
                width=env.IMAGE_WIDTH,
            )
            mappings.append((px, py, depth_value, coords))
            valid_count += 1
            msg = (f"Pixel ({px:3d}, {py:3d}) | Depth: {depth_value:.4f} "
                   f"-> World: [{coords[0]:7.4f}, {coords[1]:7.4f}, "
                   f"{coords[2]:7.4f}]")
            print(msg)
        else:
            msg = (f"Pixel ({px:3d}, {py:3d}) | Depth: {depth_value:.4f} "
                   f"-> INVALID DEPTH (skipping)")
            print(msg)

    print("\n" + "-" * 60)
    print("Analysis:")
    print("-" * 60)
    print(f"Valid mappings: {valid_count} / 10")

    if valid_count > 0:
        # Extract world coordinates
        world_coords = np.array([m[3] for m in mappings])

        # Check for unique world coordinates
        unique_coords = np.unique(world_coords[:, :2], axis=0)  # Only x, y
        print(f"Unique (x, y) world coordinates: "
              f"{len(unique_coords)} / {valid_count}")

        if len(unique_coords) < valid_count:
            print("\n⚠️  WARNING: Multiple pixels map to the same "
                  "world coordinates!")
            print("This indicates a problem with the pixel-to-world mapping.")

            # Find duplicates
            print("\nDuplicate mappings:")
            for i, (px1, py1, d1, c1) in enumerate(mappings):
                for j, (px2, py2, d2, c2) in enumerate(
                        mappings[i + 1:], start=i + 1):
                    if np.allclose(c1[:2], c2[:2], atol=1e-4):
                        msg = (f"  Pixel ({px1:3d}, {py1:3d}) and "
                               f"({px2:3d}, {py2:3d}) both map to "
                               f"[{c1[0]:.4f}, {c1[1]:.4f}]")
                        print(msg)
        else:
            print("✓ All pixels map to unique world coordinates.")

        # Show spread of coordinates
        x_coords = world_coords[:, 0]
        y_coords = world_coords[:, 1]
        z_coords = world_coords[:, 2]

        print("\nWorld coordinate ranges:")
        x_span = x_coords.max() - x_coords.min()
        print(f"  X: [{x_coords.min():.4f}, {x_coords.max():.4f}] "
              f"(span: {x_span:.4f})")
        y_span = y_coords.max() - y_coords.min()
        print(f"  Y: [{y_coords.min():.4f}, {y_coords.max():.4f}] "
              f"(span: {y_span:.4f})")
        z_span = z_coords.max() - z_coords.min()
        print(f"  Z: [{z_coords.min():.4f}, {z_coords.max():.4f}] "
              f"(span: {z_span:.4f})")

        # Check if coordinates are within expected workspace
        workspace_bounds = env.workspace_bounds
        x_min, x_max = workspace_bounds["x"]
        y_min, y_max = workspace_bounds["y"]

        in_workspace = np.sum(
            ((x_coords >= x_min) & (x_coords <= x_max)
             & (y_coords >= y_min) & (y_coords <= y_max)))
        print("\nCoordinates within workspace bounds:")
        print(f"  Workspace: x=[{x_min:.2f}, {x_max:.2f}], "
              f"y=[{y_min:.2f}, {y_max:.2f}]")
        print(f"  In workspace: {in_workspace} / {valid_count}")

    print("\n" + "=" * 60)

    # Additional test: systematic grid sampling
    print("\nAdditional test: Systematic grid sampling (5x5 grid)")
    print("-" * 60)

    grid_mappings = []
    for y_idx in range(5):
        for x_idx in range(5):
            px = int((x_idx + 0.5) * env.IMAGE_WIDTH / 5)
            py = int((y_idx + 0.5) * env.IMAGE_HEIGHT / 5)

            depth_value = depth[py, px]
            is_valid = depth_value > 0 and depth_value <= 2.0

            if is_valid:
                coords = env.controller.pixel_2_world(
                    pixel_x=px,
                    pixel_y=py,
                    depth=depth_value,
                    height=env.IMAGE_HEIGHT,
                    width=env.IMAGE_WIDTH,
                )
                grid_mappings.append((px, py, depth_value, coords))

    if grid_mappings:
        grid_coords = np.array([m[3] for m in grid_mappings])
        grid_unique = np.unique(grid_coords[:, :2], axis=0)
        print(f"Grid samples: {len(grid_mappings)} valid")
        print(f"Unique (x, y) coordinates: "
              f"{len(grid_unique)} / {len(grid_mappings)}")

        if len(grid_unique) < len(grid_mappings):
            print("⚠️  WARNING: Grid sampling also shows duplicate "
                  "mappings!")
        else:
            print("✓ Grid sampling shows unique mappings.")

    # Test specific pixels that were reported as problematic
    print("\n" + "=" * 60)
    print("Testing specific problematic pixels:")
    print("-" * 60)
    problematic_pixels = [(68, 51), (25, 127)]
    for px, py in problematic_pixels:
        depth_value = depth[py, px]
        is_valid = depth_value > 0 and depth_value <= 2.0
        if is_valid:
            coords = env.controller.pixel_2_world(
                pixel_x=px,
                pixel_y=py,
                depth=depth_value,
                height=env.IMAGE_HEIGHT,
                width=env.IMAGE_WIDTH,
            )
            print(f"Pixel ({px:3d}, {py:3d}) | Depth: {depth_value:.4f} "
                  f"-> World: [{coords[0]:7.4f}, {coords[1]:7.4f}, "
                  f"{coords[2]:7.4f}]")
        else:
            print(f"Pixel ({px:3d}, {py:3d}) | Depth: {depth_value:.4f} "
                  f"-> INVALID DEPTH")

    # Test with same depth value to see if that causes duplicates
    print("\n" + "-" * 60)
    print("Testing if same depth causes duplicate mappings:")
    print("-" * 60)
    # Find a common depth value
    depth_counts = {}
    for y in range(env.IMAGE_HEIGHT):
        for x in range(env.IMAGE_WIDTH):
            d = depth[y, x]
            if 0 < d <= 2.0:
                d_rounded = round(d, 2)  # Round to 2 decimal places
                if d_rounded not in depth_counts:
                    depth_counts[d_rounded] = []
                depth_counts[d_rounded].append((x, y))

    # Find depth values that appear frequently
    common_depths = [(d, pixels) for d, pixels in depth_counts.items()
                     if len(pixels) > 1]
    common_depths.sort(key=lambda x: len(x[1]), reverse=True)

    if common_depths:
        # Test first few pixels with the most common depth
        test_depth, test_pixels = common_depths[0]
        print(f"Testing depth value: {test_depth:.2f} "
              f"(appears {len(test_pixels)} times)")
        test_coords = []
        for px, py in test_pixels[:5]:  # Test first 5 pixels
            coords = env.controller.pixel_2_world(
                pixel_x=px,
                pixel_y=py,
                depth=test_depth,
                height=env.IMAGE_HEIGHT,
                width=env.IMAGE_WIDTH,
            )
            test_coords.append((px, py, coords))
            print(f"  Pixel ({px:3d}, {py:3d}) -> "
                  f"World: [{coords[0]:7.4f}, {coords[1]:7.4f}, "
                  f"{coords[2]:7.4f}]")

        # Check for duplicates
        unique_test = np.unique([c[2][:2] for c in test_coords], axis=0)
        if len(unique_test) < len(test_coords):
            print("\n⚠️  WARNING: Same depth value produced duplicate "
                  "world coordinates!")
        else:
            print("\n✓ Same depth value produces unique coordinates "
                  "(as expected - different pixels should map differently)")

    env.close()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_pixel_to_world_mapping()
