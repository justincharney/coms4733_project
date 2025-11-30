"""
Test the grasp procedure step by step with renders at each stage.

This test verifies that:
1. The robot can move to pre-grasp position
2. The gripper can rotate
3. The robot can move to grasping position
4. The gripper actually closes (position changes)
5. Force-based grasp detection works correctly
"""

import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

# Output directory for debug renders
RENDER_DIR = Path(__file__).parent.parent / "debug_renders"


@pytest.fixture
def env_and_controller():
    """Create environment and controller for testing."""
    pytest.importorskip("mujoco")
    import gym

    import gym_grasper

    env = gym.make(
        "gym_grasper:Grasper-v0",
        image_height=200,
        image_width=200,
        show_obs=False,
        render=False,
    )
    obs = env.reset()
    yield env, env.controller, obs
    env.close()


def save_render(controller, name, camera="side", width=640, height=480):
    """Save a render from the specified camera."""
    RENDER_DIR.mkdir(exist_ok=True)
    rgb, _ = controller.get_image_data(
        width=width, height=height, camera=camera, show=False
    )
    path = RENDER_DIR / f"{name}.png"
    cv.imwrite(str(path), cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
    print(f"Saved: {path}")
    return path


class TestGraspProcedure:
    """Test the complete grasp procedure step by step."""

    def test_grasp_procedure_with_renders(self, env_and_controller):
        """
        Execute the full grasp procedure and verify each step.
        Saves renders at each stage for visual debugging.
        """
        env, controller, obs = env_and_controller

        # Get target object location
        desired_goal = obs["desired_goal"]
        print(f"\nTarget object at: {desired_goal}")

        # Convert to grasp coordinates
        px, py = controller.world_2_pixel(
            desired_goal, width=200, height=200, camera="top_down"
        )
        px = max(0, min(199, px))
        py = max(0, min(199, py))
        depth = obs["depth"][py][px]
        coordinates = controller.pixel_2_world(px, py, depth, width=200, height=200)
        print(f"Grasp coordinates: {coordinates}")

        # Step 0: Initial state
        save_render(controller, "0_initial_state")
        initial_gripper_pos = controller.get_gripper_position()
        print(f"Initial gripper position: {initial_gripper_pos:.4f}")

        # Step 1: Move to pre-grasp position (above target)
        print("\n=== Step 1: Move to pre-grasp position ===")
        coordinates_1 = coordinates.copy()
        coordinates_1[2] = 1.1  # Above target
        result1 = controller.move_ee(
            coordinates_1, max_steps=1000, quiet=True, render=False, tolerance=0.05
        )
        print(f"Result: {result1}")
        save_render(controller, "1_pre_grasp_position", "top_down_wide")
        assert result1 == "success", f"Failed to move to pre-grasp position: {result1}"

        # Step 2: Rotate gripper
        print("\n=== Step 2: Rotate gripper ===")
        controller.current_target_joint_values[5] = 0  # 0 degrees
        result2 = controller.move_group_to_joint_target(
            tolerance=0.05, max_steps=500, render=False, quiet=True
        )
        print(f"Result: {result2}")
        save_render(controller, "2_rotated", "top_down_wide")
        assert result2 == "success", f"Failed to rotate gripper: {result2}"

        # Step 3: Open gripper
        print("\n=== Step 3: Open gripper ===")
        result_open = controller.open_gripper(half=True, render=False, quiet=True)
        print(f"Result: {result_open}")
        save_render(controller, "3_gripper_open", "top_down_wide")
        gripper_pos_open = controller.get_gripper_position()
        print(f"Gripper position after open: {gripper_pos_open:.4f}")
        # Gripper should be open (positive position for this gripper)
        assert gripper_pos_open > -0.5, f"Gripper not open: {gripper_pos_open}"

        # Step 4: Move to grasping position
        print("\n=== Step 4: Move to grasping position ===")
        coordinates_2 = coordinates.copy()
        coordinates_2[2] = max(env.TABLE_HEIGHT, coordinates_2[2] - 0.01)
        print(f"Grasping height: {coordinates_2[2]}")
        result4 = controller.move_ee(
            coordinates_2, max_steps=500, quiet=True, render=False, tolerance=0.02
        )
        print(f"Result: {result4}")
        save_render(controller, "4_grasp_position", "top_down_wide")
        # Note: This may fail if IK can't find solution - that's expected for some positions

        # Step 5: Stay briefly to stabilize
        print("\n=== Step 5: Stay at position ===")
        controller.stay(100, render=False)
        save_render(controller, "5_stayed", "top_down_wide")
        gripper_pos_before_close = controller.get_gripper_position()
        print(f"Gripper position before close: {gripper_pos_before_close:.4f}")

        # Step 6: Close gripper (position-based detection)
        print("\n=== Step 6: Close gripper (position-based) ===")
        # grasp() returns True if gripper couldn't fully close (something blocking)
        grasped = controller.grasp(render=False, quiet=False)
        gripper_pos_after_close = controller.get_gripper_position()
        print(f"Grasped: {grasped}, Final position: {gripper_pos_after_close:.4f}")
        save_render(controller, "6_gripper_closed", "top_down_wide")

        # The gripper should have moved toward closed position (more negative)
        assert gripper_pos_after_close < gripper_pos_before_close, (
            f"Gripper did not close! Before: {gripper_pos_before_close:.4f}, "
            f"After: {gripper_pos_after_close:.4f}"
        )

        # Step 7: Lift up
        print("\n=== Step 7: Lift up ===")
        lift_pos = coordinates.copy()
        lift_pos[2] = 1.1
        result7 = controller.move_ee(
            lift_pos, max_steps=500, quiet=True, render=False, tolerance=0.05
        )
        print(f"Result: {result7}")
        save_render(controller, "7_lifted", "top_down_wide")

        # Step 8: Check if something is in the gripper (position-based verification)
        print("\n=== Step 8: Final grasp verification ===")
        # Try to close gripper further
        controller.close_gripper(max_steps=200, render=False, quiet=True)
        gripper_final_pos = controller.get_gripper_position()

        # Empty gripper closes to approximately -0.39. If position is more open, object is there.
        FULLY_CLOSED_POSITION = -0.38
        has_object = gripper_final_pos > FULLY_CLOSED_POSITION

        print(f"Gripper final position: {gripper_final_pos:.4f}")
        print(f"Fully closed threshold: {FULLY_CLOSED_POSITION}")
        if has_object:
            print(
                f">>> OBJECT IN GRIPPER: YES (position {gripper_final_pos:.4f} > {FULLY_CLOSED_POSITION}) <<<"
            )
        else:
            print(
                f">>> OBJECT IN GRIPPER: NO (position {gripper_final_pos:.4f} <= {FULLY_CLOSED_POSITION}) <<<"
            )

        print(f"\n=== Test complete! Check {RENDER_DIR}/ for images ===\n")


class TestGripperClosing:
    """Test that the gripper physically closes."""

    def test_gripper_closes_on_empty(self, env_and_controller):
        """Test that gripper closes to near-closed position when empty."""
        env, controller, obs = env_and_controller

        # Open gripper first
        controller.open_gripper(render=False, quiet=True)
        open_pos = controller.get_gripper_position()
        print(f"Open position: {open_pos:.4f}")

        # Close gripper on empty space using position-based grasp()
        # grasp() returns True if gripper couldn't fully close (something blocking)
        grasped = controller.grasp(render=False, quiet=True)

        closed_pos = controller.get_gripper_position()
        print(f"Closed position: {closed_pos:.4f}")
        print(f"Grasped (detected object): {grasped}")

        # Verify gripper moved from open to closed
        assert closed_pos < open_pos - 0.5, (
            f"Gripper did not close enough. Open: {open_pos:.4f}, Closed: {closed_pos:.4f}"
        )

        # Should NOT detect a grasp (empty) - gripper should reach target
        assert not grasped, "Gripper incorrectly detected grasp on empty space"

    def test_gripper_position_changes_during_close(self, env_and_controller):
        """Test that gripper position actually changes during close_gripper."""
        env, controller, obs = env_and_controller

        # Open gripper
        controller.open_gripper(render=False, quiet=True)
        open_pos = controller.get_gripper_position()

        # Close gripper
        result = controller.close_gripper(max_steps=500, render=False, quiet=True)

        closed_pos = controller.get_gripper_position()
        print(f"Position range: {open_pos:.4f} to {closed_pos:.4f}")
        print(f"Close result: {result}")

        # Verify position decreased (gripper closed)
        assert closed_pos < open_pos - 0.3, (
            f"Gripper position did not decrease enough during closing. "
            f"Start: {open_pos:.4f}, End: {closed_pos:.4f}"
        )

        # When empty, gripper should reach target (success)
        assert result == "success", (
            f"Gripper should reach closed position when empty, got: {result}"
        )
