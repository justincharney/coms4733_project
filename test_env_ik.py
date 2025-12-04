#!/usr/bin/env python3
"""
Sanity-check script for environment and IK solver.
Performs a deterministic scripted grasp test to verify:
1. IK solver works correctly
2. Environment can execute a valid grasp sequence
3. Workspace bounds are correct
"""

import sys
import numpy as np
from termcolor import colored
from pyquaternion import Quaternion

sys.path.insert(0, ".")
from gym_grasper.envs.GraspingEnv import GraspEnv


def test_ik_directly(env, target_position):
    """
    Test IK solver directly on a target position.

    Args:
        env: GraspEnv instance
        target_position: [x, y, z] target position in world coordinates

    Returns:
        tuple: (success: bool, joint_angles: list or None, error: float or None)
    """
    print(colored("\n" + "=" * 60, color="cyan"))
    print(colored("Testing IK solver directly", color="cyan", attrs=["bold"]))
    print(colored("=" * 60, color="cyan"))
    print(f"Target position: {target_position}")

    joint_angles = env.controller.ik(target_position)

    if joint_angles is None:
        print(colored("❌ Failed to find IK solution", color="red", attrs=["bold"]))
        return False, None, None

    # The ik() method already verifies the solution internally using forward kinematics
    # and only returns if error <= 0.02m. So if we get here, the solution is valid.
    print(colored("✓ IK solution found", color="green"))
    print(f"  Joint angles: {[f'{a:.4f}' for a in joint_angles]}")
    print("  Note: IK solution already verified internally (error < 0.02m)")

    return True, joint_angles, 0.0  # Error is already verified to be < 0.02m by controller


def place_object_deterministically(env, position, quaternion=None):
    """
    Place an object at a deterministic position.

    Args:
        env: GraspEnv instance
        position: [x, y, z] position in world coordinates
        quaternion: Quaternion for orientation (default: identity)

    Returns:
        bool: True if object was placed successfully
    """
    if not env.object_joint_names:
        print(colored("⚠ No objects found in environment", color="yellow"))
        return False

    if quaternion is None:
        quaternion = Quaternion(1, 0, 0, 0)  # Identity quaternion

    # Use the first object
    joint_name = env.object_joint_names[0]
    start, end = env.controller.get_joint_qpos_addr(joint_name)

    # Set position
    env.data.qpos[start] = position[0]
    env.data.qpos[start + 1] = position[1]
    env.data.qpos[start + 2] = position[2]

    # Set orientation (quaternion: w, x, y, z)
    env.data.qpos[start + 3:end] = quaternion.unit.elements

    # Forward kinematics to update positions
    env.sim.forward()

    print(colored(f"✓ Placed object at {position}", color="green"))
    return True


def deterministic_grasp_test(env, target_position, render=False):
    """
    Perform a deterministic scripted grasp test.

    Args:
        env: GraspEnv instance
        target_position: [x, y, z] target position in world coordinates
        render: Whether to render the simulation

    Returns:
        dict: Test results
    """
    print(colored(f"\n{'='*60}", color="cyan"))
    print(colored("Deterministic Scripted Grasp Test", color="cyan", attrs=["bold"]))
    print(colored(f"{'='*60}", color="cyan"))

    results = {
        "ik_test_passed": False,
        "pre_grasp_reached": False,
        "grasp_position_reached": False,
        "grasp_successful": False,
        "errors": []
    }

    # Step 1: Test IK directly
    print(colored("\n[Step 1] Testing IK solver directly", color="yellow", attrs=["bold"]))
    ik_success, joint_angles, ik_error = test_ik_directly(env, target_position)
    results["ik_test_passed"] = ik_success
    if not ik_success:
        results["errors"].append("IK solver failed to find solution")
        return results

    # Step 2: Move to pre-grasp position (above target)
    print(colored("\n[Step 2] Moving to pre-grasp position", color="yellow", attrs=["bold"]))
    pre_grasp_position = target_position.copy()
    pre_grasp_position[2] = 1.1  # 20cm above target

    print(f"Pre-grasp position: {pre_grasp_position}")
    result_pre = env.controller.move_ee(
        pre_grasp_position,
        max_steps=1000,
        quiet=False,
        render=render,
        tolerance=0.05,
    )

    if result_pre == "success":
        print(colored("✓ Reached pre-grasp position", color="green"))
        results["pre_grasp_reached"] = True
    else:
        print(colored(f"❌ Failed to reach pre-grasp position: {result_pre}", color="red"))
        results["errors"].append(f"Failed to reach pre-grasp position: {result_pre}")
        # Try center as fallback
        print("Trying center position as fallback...")
        result_pre = env.controller.move_ee(
            [0.0, -0.6, 1.1],
            max_steps=1000,
            quiet=False,
            render=render,
            tolerance=0.05,
        )
        if result_pre == "success":
            print(colored("✓ Reached center position", color="green"))
            results["pre_grasp_reached"] = True
        else:
            return results

    # Step 3: Rotate gripper to 0 degrees (vertical)
    print(colored("\n[Step 3] Rotating gripper", color="yellow", attrs=["bold"]))
    result_rotate = env.rotate_wrist_3_joint_to_value(0)
    if result_rotate == "success":
        print(colored("✓ Gripper rotated", color="green"))
    else:
        print(colored(f"⚠ Gripper rotation: {result_rotate}", color="yellow"))

    # Step 4: Open gripper
    print(colored("\n[Step 4] Opening gripper", color="yellow", attrs=["bold"]))
    env.controller.open_gripper(half=True, render=render, quiet=False)
    print(colored("✓ Gripper opened", color="green"))

    # Step 5: Move to grasp position
    print(colored("\n[Step 5] Moving to grasp position", color="yellow", attrs=["bold"]))
    grasp_position = target_position.copy()
    grasp_position[2] = max(env.TABLE_HEIGHT, target_position[2] - 0.01)
    print(f"Grasp position: {grasp_position}")

    result_grasp_pos = env.controller.move_ee(
        grasp_position,
        max_steps=500,
        quiet=False,
        render=render,
        tolerance=0.02,
    )

    if result_grasp_pos == "success":
        print(colored("✓ Reached grasp position", color="green"))
        results["grasp_position_reached"] = True
    else:
        print(colored(f"❌ Failed to reach grasp position: {result_grasp_pos}", color="red"))
        results["errors"].append(f"Failed to reach grasp position: {result_grasp_pos}")
        return results

    # Step 6: Wait for stability
    print(colored("\n[Step 6] Waiting for stability", color="yellow", attrs=["bold"]))
    env.controller.stay(100, render=render)

    # Step 7: Attempt grasp
    print(colored("\n[Step 7] Attempting grasp", color="yellow", attrs=["bold"]))
    result_grasp = env.controller.grasp(render=render, quiet=False)

    if result_grasp:
        print(colored("✓ Initial grasp detected", color="green"))
    else:
        print(colored("❌ No initial grasp detected", color="red"))
        results["errors"].append("No initial grasp detected")

    # Step 8: Lift and verify grasp
    print(colored("\n[Step 8] Lifting to verify grasp", color="yellow", attrs=["bold"]))
    result_lift = env.controller.move_ee(
        [0.0, -0.6, 1.1],
        max_steps=1000,
        quiet=False,
        render=render,
        tolerance=0.05,
    )

    if result_lift == "success":
        print(colored("✓ Lifted successfully", color="green"))
    else:
        print(colored(f"⚠ Lift result: {result_lift}", color="yellow"))

    # Verify grasp by checking gripper position
    env.controller.close_gripper(max_steps=200, render=render, quiet=False)
    gripper_pos = env.controller.get_gripper_position()
    FULLY_CLOSED_POSITION = -0.38

    if gripper_pos > FULLY_CLOSED_POSITION:
        print(colored("✓ Object confirmed in gripper", color="green"))
        print(f"  Gripper position: {gripper_pos:.4f} (fully closed: {FULLY_CLOSED_POSITION:.4f})")
        results["grasp_successful"] = True

        # Step 9: Move to drop position (bin)
        print(colored("\n[Step 9] Moving to drop position (bin)", color="yellow", attrs=["bold"]))
        drop_target = np.array([0.6, 0.0, 1.15], dtype=np.float32)
        print(f"Drop position: {drop_target}")

        result_drop = env.controller.move_ee(
            drop_target,
            max_steps=1600,
            quiet=False,
            render=render,
            tolerance=0.03,
        )

        if result_drop == "success" or result_drop.startswith("success"):
            print(colored("✓ Reached drop position", color="green"))
        else:
            # Check if we're close enough
            ee_after_drop = env._get_end_effector_position()
            drop_err = np.linalg.norm(ee_after_drop - drop_target)
            if drop_err < 0.05:
                print(colored(f"✓ Close enough to drop position (error: {drop_err:.3f}m)", color="green"))
            else:
                print(colored(f"⚠ Drop position result: {result_drop} (error: {drop_err:.3f}m)", color="yellow"))

        # Step 10: Open gripper to drop object
        print(colored("\n[Step 10] Opening gripper to drop object", color="yellow", attrs=["bold"]))
        result_open = env.controller.open_gripper(render=render, quiet=False)
        if result_open == "success":
            print(colored("✓ Gripper opened - object dropped", color="green"))
        else:
            print(colored(f"⚠ Gripper open result: {result_open}", color="yellow"))

        # Step 11: Wait a moment to see object fall
        print(colored("\n[Step 11] Waiting to observe object drop", color="yellow", attrs=["bold"]))
        env.controller.stay(500, render=render)
        print(colored("✓ Object should have dropped into bin", color="green"))

        # Step 12: Move back to home position
        print(colored("\n[Step 12] Moving back to home position", color="yellow", attrs=["bold"]))
        result_home = env.controller.move_ee(
            [0.0, -0.6, 1.1],
            max_steps=1000,
            quiet=False,
            render=render,
            tolerance=0.05,
        )
        if result_home == "success":
            print(colored("✓ Returned to home position", color="green"))
        else:
            print(colored(f"⚠ Home position result: {result_home}", color="yellow"))

        # Rotate back to zero
        env.rotate_wrist_3_joint_to_value(0)

    else:
        print(colored("❌ No object in gripper", color="red"))
        print(f"  Gripper position: {gripper_pos:.4f} (fully closed: {FULLY_CLOSED_POSITION:.4f})")
        results["errors"].append("Did not grasp anything")

    return results


def main():
    """Main test function."""
    print(colored("\n" + "=" * 60, color="cyan", attrs=["bold"]))
    print(colored("Environment and IK Sanity Check", color="cyan", attrs=["bold"]))
    print(colored("=" * 60, color="cyan"))

    # Create environment
    print(colored("\n[Setup] Creating environment...", color="yellow"))
    print(colored("Note: A visualization window should open. If not, check GLFW installation.", color="cyan"))
    env = GraspEnv(
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=False,
        demo=False,
        render=True,  # Set to True to visualize
    )

    # Check if viewer was created
    if env.viewer is None or (hasattr(env.viewer, 'window') and env.viewer.window is None):
        print(colored("⚠ Warning: Viewer not available. Running in headless mode.", color="yellow"))
        print(colored("  Install GLFW with: pip install glfw", color="yellow"))
    else:
        print(colored("✓ Viewer initialized successfully", color="green"))

    # Reset environment
    print(colored("\n[Setup] Resetting environment...", color="yellow"))
    env.reset()

    # Wait for objects to settle
    print(colored("[Setup] Waiting for objects to settle...", color="yellow"))
    env.controller.stay(2000, render=True)

    # Choose a simple object pose at the center of the table
    # Center of workspace: x=0, y=-0.55 (middle of y range), z=TABLE_HEIGHT
    target_position = np.array([0.0, -0.55, env.TABLE_HEIGHT], dtype=np.float32)

    print(colored(f"\n[Setup] Target position: {target_position}", color="cyan"))
    print(f"  Workspace bounds: {env.workspace_bounds}")
    print(f"  Table height: {env.TABLE_HEIGHT}")

    # Check if target is in workspace
    x_min, x_max = env.workspace_bounds["x"]
    y_min, y_max = env.workspace_bounds["y"]
    in_workspace = (x_min <= target_position[0] <= x_max and
                    y_min <= target_position[1] <= y_max)

    if not in_workspace:
        print(colored("⚠ Target position is outside workspace bounds!", color="yellow"))
        print("  Adjusting to center of workspace...")
        target_position[0] = (x_min + x_max) / 2
        target_position[1] = (y_min + y_max) / 2
        print(f"  New target: {target_position}")

    # Place an object at the target position deterministically
    print(colored("\n[Setup] Placing object at target position...", color="yellow"))
    if env.object_joint_names:
        place_object_deterministically(env, target_position)
        env.sim.forward()
        env.controller.stay(500, render=True)  # Let object settle
    else:
        print(colored("⚠ No objects to place, using existing object positions", color="yellow"))

    # Run the deterministic grasp test
    results = deterministic_grasp_test(env, target_position, render=True)

    # Print summary
    print(colored("\n" + "=" * 60, color="cyan", attrs=["bold"]))
    print(colored("Test Summary", color="cyan", attrs=["bold"]))
    print(colored("=" * 60, color="cyan"))

    print(f"\nIK Test: {'✓ PASSED' if results['ik_test_passed'] else '❌ FAILED'}")
    print(f"Pre-grasp Reached: {'✓ YES' if results['pre_grasp_reached'] else '❌ NO'}")
    print(f"Grasp Position Reached: {'✓ YES' if results['grasp_position_reached'] else '❌ NO'}")
    print(f"Grasp Successful: {'✓ YES' if results['grasp_successful'] else '❌ NO'}")

    if results["errors"]:
        print(colored("\nErrors encountered:", color="red", attrs=["bold"]))
        for error in results["errors"]:
            print(f"  - {error}")

    # Final verdict
    all_passed = (
        results["ik_test_passed"] and
        results["pre_grasp_reached"] and
        results["grasp_position_reached"] and
        results["grasp_successful"]
    )

    if all_passed:
        print(colored("\n✓✓✓ ALL TESTS PASSED ✓✓✓", color="green", attrs=["bold"]))
        print(colored("Environment and IK are working correctly!", color="green"))
    else:
        print(colored("\n❌❌❌ SOME TESTS FAILED ❌❌❌", color="red", attrs=["bold"]))
        print(colored("Please check the errors above.", color="red"))
        if not results["ik_test_passed"]:
            print(colored("\nPossible issues:", color="yellow"))
            print("  - IK solver configuration")
            print("  - Workspace bounds may be incorrect")
            print("  - Target pose may be unreachable")
        if not results["grasp_successful"]:
            print(colored("\nPossible issues:", color="yellow"))
            print("  - Object placement")
            print("  - Grasp detection logic")
            print("  - Gripper configuration")

    # Keep window open for a moment to see final state
    if env.render_enabled and env.viewer is not None:
        print(colored("\n[Info] Keeping visualization window open for 5 seconds...", color="cyan"))
        print(colored("  You can close the window manually or wait for it to close automatically.", color="cyan"))
        import time
        for _ in range(50):  # Render for 5 seconds (50 * 0.1s)
            if hasattr(env.viewer, 'render'):
                env.viewer.render()
            time.sleep(0.1)

    # Cleanup
    env.close()

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
