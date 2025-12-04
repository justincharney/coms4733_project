#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)
import sys

sys.path.insert(0, "..")
import copy
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import gym
import mujoco as mj
import numpy as np
from gym import spaces, utils
from pyquaternion import Quaternion
from termcolor import colored

from gym_grasper.controller.MujocoController import (
    MJ_Controller,
    MjSimWrapper,
    MuJoCoViewer,
)


class GraspEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=True,
        demo=False,
        render=False,
    ):
        self.initialized = False
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rotations = {0: 0, 1: 30, 2: 60, 3: 90, 4: -30, 5: -60}
        self.action_space_type = "multidiscrete"
        self.step_called = 0
        self.goal_tolerance = 0.03
        self.desired_goal = np.zeros(3, dtype=np.float32)
        self.last_achieved_goal = np.zeros(3, dtype=np.float32)
        self.goal_joint_name = None
        self.target_body_id = None
        self.object_joint_names = []
        self.joint_to_body = {}
        self.last_grasped_object_pose = None
        self.last_grasped_object_name = None
        self.unreachable_goal = False
        self.unreachable_goal_count = 0
        # Conservative workspace limits shared with the agent's action mask.
        # Using tightened bounds to avoid IK failures at edges.
        self.workspace_bounds = {"x": (-0.25, 0.25), "y": (-0.70, -0.40)}
        utils.EzPickle.__init__(
            self, file, image_width, image_height, show_obs, demo, render
        )
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        full_path = path + file
        self.model = mj.MjModel.from_xml_path(full_path)
        self.sim = MjSimWrapper(self.model)
        self.data = self.sim.data
        self.frame_skip = 1
        self.dt = self.model.opt.timestep * self.frame_skip
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1 / self.dt)),
        }
        # Create proper viewer when rendering is enabled, otherwise use None
        if render:
            self.viewer = MuJoCoViewer(self.model, self.data)
        else:
            self.viewer = None
        self.controller = MJ_Controller(
            self.model, self.sim, self.viewer if render else False
        )
        self.initialized = True
        self.grasp_counter = 0
        self.show_observations = show_obs
        self.demo_mode = demo
        self.TABLE_HEIGHT = 0.91
        self.render_enabled = render
        self._cache_object_metadata()
        self._set_action_space()
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
                    dtype=np.uint8,
                ),
                "depth": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
                    dtype=np.float32,
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )

    def __repr__(self):
        return f"GraspEnv(obs height={self.IMAGE_HEIGHT}, obs_width={self.IMAGE_WIDTH}, AS={self.action_space_type})"

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = utils.seeding.np_random(seed)
        self.step_called = 0
        observation = self.reset_model(show_obs=self.show_observations)
        self.current_observation = observation

        # Abort immediately if no reachable goal could be sampled
        if self.unreachable_goal:
            print(
                colored(
                    "[Workspace] Aborting episode immediately - no reachable goal found at reset",
                    color="red",
                    attrs=["bold"],
                )
            )

        return observation

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mj.mj_forward(self.model, self.data)

    def render(self, mode="human"):
        rgb, _ = self.controller.get_image_data(
            width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=False
        )
        if mode == "rgb_array":
            return rgb
        return None

    def step(self, action, record_grasps=False, markers=False, action_info="no info"):
        """
        Lets the agent execute the action.

        Args:
            action: The action to be performed.

        Returns:
            observation: np-array containing the camera image data
            rewards: The reward obtained
            done: Flag indicating weather the episode has finished or not
            info: Extra info
        """

        done = False
        info = {}
        her_reward = 0.0
        grasped_something = False
        goal_before_action = self.desired_goal.copy()
        achieved_goal = self.last_achieved_goal.copy()
        if achieved_goal.shape[0] == 0:
            achieved_goal = np.zeros(3, dtype=np.float32)
        if self.unreachable_goal:
            # Abort the episode immediately if we could not sample a reachable goal at reset.
            self.unreachable_goal = False
            self.unreachable_goal_count += 1
            print(
                colored(
                    f"[Workspace] Truncating episode; unreachable goal (count={self.unreachable_goal_count})",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            reward = -1.0
            done = True
            self.step_called += 1
            info["desired_goal"] = goal_before_action.copy()
            info["achieved_goal"] = achieved_goal.copy()
            info["truncated"] = True
            info["unreachable_goal"] = True
            info["grasp_success"] = 0.0
            info["is_success"] = 0.0
            info["her_reward"] = her_reward
            return self.current_observation, reward, done, info
        # Parent class will step once during init to set up the observation space,
        # controller is not yet available at that time.
        # Therefore we simply return a dictionary of zeros of the appropriate size.
        if not self.initialized:
            self.current_observation = defaultdict()
            self.current_observation["rgb"] = np.zeros(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
            )
            self.current_observation["depth"] = np.zeros(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            )
            self.current_observation["desired_goal"] = self.desired_goal.copy()
            self.current_observation["achieved_goal"] = self.last_achieved_goal.copy()
            self.current_observation["action_mask"] = np.ones(
                self.IMAGE_HEIGHT * self.IMAGE_WIDTH, dtype=bool
            )  # Default: all pixels reachable during init
            reward = 0
        else:
            # Use the observation from the end of the previous step
            # (or from reset if this is the first step after reset)
            # The observation is updated at the end of each step to ensure it reflects
            # the current state after the previous action
            if self.step_called == 0:
                # First step after reset - observation should already be set by reset()
                # But ensure we have a fresh one if needed
                if not hasattr(self, 'current_observation') or self.current_observation is None:
                    self.current_observation = self.get_observation(show=False)
            self._refresh_desired_goal()
            goal_before_action = self.desired_goal.copy()
            achieved_goal = self._get_end_effector_position()

            # If goal slipped out of workspace, select a new goal instead of truncating
            if not self._goal_in_workspace(goal_before_action):
                goal_ok = self._set_new_goal(self.data.qpos)
                if goal_ok:
                    # Successfully selected a new goal, refresh and continue
                    self._refresh_desired_goal()
                    goal_before_action = self.desired_goal.copy()
                else:
                    # No valid goals remain, truncate the episode
                    self.unreachable_goal_count += 1
                    reward = -1.0
                    done = True
                    self.step_called += 1
                    info["desired_goal"] = goal_before_action.copy()
                    info["achieved_goal"] = achieved_goal.copy()
                    info["truncated"] = True
                    info["unreachable_goal"] = True
                    info["grasp_success"] = 0.0
                    info["is_success"] = 0.0
                    info["her_reward"] = 0.0
                    return self.current_observation, reward, done, info
            self.last_grasped_object_pose = None
            self.last_grasped_object_name = None

            if self.action_space_type == "discrete":
                x = action % self.IMAGE_WIDTH
                y = action // self.IMAGE_WIDTH

            elif self.action_space_type == "multidiscrete":
                x = action[0] % self.IMAGE_WIDTH
                y = action[0] // self.IMAGE_WIDTH
                rotation = action[1]

            # Depth value for the pixel corresponding to the action
            depth = self.current_observation["depth"][y][x]

            # Validate depth value BEFORE conversion to catch invalid pixels early
            if depth <= 0.0 or depth > 2.0:
                msg = ("Action ({}): Pixel X: {}, Pixel Y: {}, Rotation: {} "
                       "({} deg), Depth: {:.4f} -> INVALID DEPTH (skipping)")
                print(
                    colored(
                        msg.format(
                            action_info, x, y, rotation,
                            self.rotations[rotation], depth
                        ),
                        color="red",
                        attrs=["bold"],
                    )
                )
                reward = -0.3
                reach_success = False
                grasp_coordinates = None
                grasped_something = False  # Ensure it's set to False for invalid depth
            else:
                coordinates = self.controller.pixel_2_world(
                    pixel_x=x,
                    pixel_y=y,
                    depth=depth,
                    height=self.IMAGE_HEIGHT,
                    width=self.IMAGE_WIDTH,
                )
                print(
                    colored(
                        "Action ({}): Pixel X: {}, Pixel Y: {}, Rotation: {} ({} deg), Depth: {:.4f}".format(
                            action_info, x, y, rotation, self.rotations[rotation], depth
                        ),
                        color="blue",
                        attrs=["bold"],
                    )
                )
                print(
                    colored(
                        "Transformed into world coordinates: [{:.4f}, {:.4f}, {:.4f}]".format(
                            coordinates[0], coordinates[1], coordinates[2]
                        ),
                        color="blue",
                        attrs=["bold"],
                    )
                )

                # Check for coordinates we don't need to try to save some time
                if coordinates[2] < 0.8 or coordinates[1] > -0.3:
                    print(
                        colored(
                            "Skipping execution due to bad coordinates!",
                            color="red",
                            attrs=["bold"],
                        )
                    )
                    # Binary reward
                    reward = -0.3
                    reach_success = False
                    grasp_coordinates = None
                else:
                    grasped_something, grasp_coordinates, reach_success = (
                        self.move_and_grasp(
                            coordinates,
                            rotation,
                            render=self.render_enabled,
                            record_grasps=record_grasps,
                            markers=markers,
                        )
                    )

            if grasped_something:
                if self.last_grasped_object_pose is None:
                    _, detected_pos = self._detect_object_close_to_gripper(
                        distance_threshold=0.08
                    )
                    if detected_pos is not None:
                        self.last_grasped_object_pose = detected_pos.copy()
                if self.last_grasped_object_pose is not None:
                    achieved_goal = self.last_grasped_object_pose.copy()
                else:
                    achieved_goal = self._get_end_effector_position()
            elif reach_success and grasp_coordinates is not None:
                # If we reached the target but failed to grasp, we still achieved the goal of reaching that location.
                # This helps HER to reinforce the reaching policy.
                achieved_goal = np.array(grasp_coordinates, dtype=np.float32)
            else:
                achieved_goal = self._get_end_effector_position()

            self.last_achieved_goal = achieved_goal
            # Compute her_reward after determining grasped_something for consistency
            her_reward = float(self.compute_reward(achieved_goal, goal_before_action, grasp_success=grasped_something))
            if grasped_something:
                reward = 5.0
            elif not reach_success:
                # Explicit penalty for IK/reach failures
                reward = -0.2
            else:
                # Distance-based shaping reward (negative) to guide the agent to the object
                reward = max(-0.5, her_reward)

            if self.initialized:
                print(
                    colored(
                        "Reward received during step: {}".format(reward),
                        color="yellow",
                        attrs=["bold"],
                    )
                )

            self.current_observation = self.get_observation(show=self.show_observations)

            if grasped_something:
                done = True

        self.step_called += 1
        info["desired_goal"] = goal_before_action.copy()
        info["achieved_goal"] = self.last_achieved_goal.copy()
        info["is_success"] = float(grasped_something)
        info["her_reward"] = her_reward
        info["grasp_success"] = 1.0 if grasped_something else 0.0

        return self.current_observation, reward, done, info

    def _set_action_space(self):
        if self.action_space_type == "discrete":
            size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
            self.action_space = spaces.Discrete(size)
        elif self.action_space_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete(
                [self.IMAGE_HEIGHT * self.IMAGE_WIDTH, len(self.rotations)]
            )

        return self.action_space

    def set_grasp_position(self, position):
        """
        Legacy method, not used in the current setup.
        May be used to directly set the joint values to a desired position.
        """

        joint_angles = self.controller.ik(position)
        qpos = self.data.qpos
        idx = self.controller.actuated_joint_ids[self.controller.groups["Arm"]]
        for i, index in enumerate(idx):
            qpos[index] = joint_angles[i]

        self.controller.set_group_joint_target(group="Arm", target=joint_angles)

        idx_2 = self.controller.actuated_joint_ids[self.controller.groups["Gripper"]]

        open_gripper_values = [0.2, 0.2, 0.0, -0.1]

        for i, index in enumerate(idx_2):
            qpos[index] = open_gripper_values[i]

        qvel = np.zeros(len(self.data.qvel))
        self.set_state(qpos, qvel)
        self.data.ctrl[:] = 0

    def rotate_wrist_3_joint_to_value(self, degrees):
        self.controller.current_target_joint_values[5] = math.radians(degrees)
        return self.controller.move_group_to_joint_target(
            tolerance=0.05, max_steps=500, render=self.render_enabled, quiet=True
        )

    def transform_height(self, height_action, depth_height):
        return np.round(
            self.TABLE_HEIGHT + height_action * (0.1) / self.action_space.nvec[1],
            decimals=3,
        )

    def move_and_grasp(
        self,
        coordinates,
        rotation,
        render=False,
        record_grasps=False,
        markers=False,
        plot=False,
    ):
        # Try to move directly above target
        coordinates_1 = copy.deepcopy(coordinates)
        coordinates_1[2] = 1.1
        result1 = self.controller.move_ee(
            coordinates_1,
            max_steps=1000,
            quiet=True,
            render=render,
            marker=markers,
            tolerance=0.05,
            plot=plot,
        )
        steps1 = self.controller.last_steps
        result_pre = "Failed"
        if result1 == "success":
            result_pre = "Above target"

        # If that's not possible, move to center as pre grasp position
        if result1[:2] == "No":
            result1 = self.controller.move_ee(
                [0.0, -0.6, 1.1],
                max_steps=1000,
                quiet=True,
                render=render,
                marker=markers,
                tolerance=0.05,
                plot=plot,
            )
            steps1 = self.controller.last_steps
            if result1 == "success":
                result_pre = "Center"

        # Rotate gripper according to second action dimension
        result_rotate = self.rotate_wrist_3_joint_to_value(self.rotations[rotation])
        steps_rotate = self.controller.last_steps

        self.controller.open_gripper(half=True, render=render, quiet=True, plot=plot)

        # Move to grasping height
        coordinates_2 = copy.deepcopy(coordinates)
        GRASP_MIN = self.TABLE_HEIGHT + 0.01
        GRASP_MAX = 1.10  # same as your pre-grasp z

        coordinates_2[2] = np.clip(self.TABLE_HEIGHT + 0.02, GRASP_MIN, GRASP_MAX)
        result2 = self.controller.move_ee(
            coordinates_2,
            max_steps=500,
            quiet=True,
            render=render,
            marker=markers,
            tolerance=0.02,
            plot=plot,
        )
        steps2 = self.controller.last_steps

        # Only attempt grasp if we reached the target position
        reach_success = result2 == "success"
        if reach_success:
            self.controller.stay(100, render=render)
            # Position-based grasp detection: if gripper can't fully close, something is blocking
            result_grasp = self.controller.grasp(render=render, quiet=True)
            if result_grasp:
                self._capture_grasp_snapshot()
        else:
            result_grasp = False

        self.controller.actuators[0][4].Kp = 10.0

        # Move back above center of table (lift)
        result3 = self.controller.move_ee(
            [0.0, -0.6, 1.1],
            max_steps=1000,
            quiet=True,
            render=render,
            plot=plot,
            marker=markers,
            tolerance=0.05,
        )
        steps3 = self.controller.last_steps

        # Verify grasp immediately after lifting (position-based check)
        # Empty gripper closes to approximately -0.39. If gripper position is
        # significantly different (more open), an object is blocking it.
        final_grasp_confirmed = False
        FULLY_CLOSED_POSITION = -0.38  # Empty gripper baseline

        if result_grasp:
            # Try to close gripper further
            self.controller.close_gripper(max_steps=200, render=render, quiet=True)
            gripper_pos = self.controller.get_gripper_position()
            # Object is in gripper if position is significantly more open than fully closed
            final_grasp_confirmed = gripper_pos > FULLY_CLOSED_POSITION

        final_str = (
            "Object in the gripper"
            if final_grasp_confirmed
            else "Nothing in the gripper"
        )
        grasped_something = final_grasp_confirmed and result_grasp

        # Move to drop position
        drop_target = np.array([0.6, 0.0, 1.15], dtype=np.float32)
        result4 = self.controller.move_ee(
            drop_target,
            max_steps=1600,
            quiet=True,
            render=render,
            plot=plot,
            marker=markers,
            tolerance=0.03,
        )
        steps4 = self.controller.last_steps

        # If we hit the step limit but are effectively at the drop pose, treat it as success
        if result4.startswith("max"):
            ee_after_drop = self._get_end_effector_position()
            drop_err = np.linalg.norm(ee_after_drop - drop_target)
            if drop_err < 0.05:
                result4 = f"success (within {drop_err:.3f} m of drop target)"
            else:
                print(f"Drop failed. Distance to target: {drop_err:.3f} m")
        if not grasped_something:
            self.last_grasped_object_pose = None
            self.last_grasped_object_name = None

        if grasped_something and record_grasps:
            capture_rgb, _ = self.controller.get_image_data(
                width=800, height=800, camera="side"
            )
            self.grasp_counter += 1
            os.makedirs("observations", exist_ok=True)
            img_name = "observations/Grasp_{}.png".format(self.grasp_counter)
            cv.imwrite(img_name, cv.cvtColor(capture_rgb, cv.COLOR_RGB2BGR))

        # Open gripper again
        result_open = self.controller.open_gripper(render=render, quiet=True, plot=plot)
        steps_open = self.controller.last_steps

        if grasped_something:
            self.controller.stay(200, render=render)

        # Move back to zero rotation
        self.rotate_wrist_3_joint_to_value(0)

        self.controller.actuators[0][4].Kp = 20.0

        print("Results: ")
        print(
            "Move to pre grasp position: ".ljust(40, " "),
            result_pre,
            ",",
            steps1,
            "steps",
        )
        print(
            "Rotate gripper: ".ljust(40, " "), result_rotate, ",", steps_rotate, "steps"
        )
        move_to_grasp_str = (
            f"Move to grasping position (z="
            f"{np.round(coordinates_2[2], decimals=4) if isinstance(coordinates_2, np.ndarray) else 0}"
            f"):"
        )
        print(
            move_to_grasp_str.ljust(40, " "),
            result2,
            ",",
            steps2,
            "steps",
        )
        print("Initial grasp detected?: ".ljust(40, " "), result_grasp)
        print("Move to center (lift): ".ljust(40, " "), result3, ",", steps3, "steps")
        print("Object still in gripper?: ".ljust(40, " "), final_str)
        print("Move to drop position: ".ljust(40, " "), result4, ",", steps4, "steps")
        print("Open gripper: ".ljust(40, " "), result_open, ",", steps_open, "steps")

        if all(
            str(r).startswith("success")
            for r in (result1, result2, result3, result4, result_open)
        ):
            print(
                colored(
                    "Executed all movements successfully.",
                    color="green",
                    attrs=["bold"],
                )
            )
        else:
            print(
                colored(
                    "Could not execute all movements successfully.",
                    color="red",
                    attrs=["bold"],
                )
            )

        if grasped_something:
            print(colored("Successful grasp!", color="green", attrs=["bold"]))
            return True, coordinates_2, reach_success
        else:
            print(colored("Did not grasp anything.", color="red", attrs=["bold"]))
            return False, coordinates_2, reach_success

    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in a cv2 window.
        """

        self._refresh_desired_goal()

        rgb, depth = self.controller.get_image_data(
            width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show
        )
        depth = self.controller.depth_2_meters(depth)

        # Diagnostic: Check if depth image is all zeros (indicates rendering issue)
        if np.all(depth == 0.0):
            print(
                colored(
                    f"[WARNING] Depth image is all zeros! Depth range: [{depth.min():.4f}, {depth.max():.4f}], "
                    f"Non-zero pixels: {np.count_nonzero(depth)} / {depth.size}",
                    color="yellow",
                    attrs=["bold"],
                )
            )
        elif np.sum((depth > 0) & (depth <= 2.0)) < depth.size * 0.1:
            # Less than 10% of pixels have valid depth
            valid_pixels = np.sum((depth > 0) & (depth <= 2.0))
            print(
                colored(
                    f"[WARNING] Very few valid depth pixels: {valid_pixels} / {depth.size} "
                    f"({100*valid_pixels/depth.size:.1f}%)",
                    color="yellow",
                    attrs=["bold"],
                )
            )

        observation = defaultdict()
        observation["rgb"] = rgb
        observation["depth"] = depth
        observation["desired_goal"] = self.desired_goal.copy()
        observation["achieved_goal"] = self.last_achieved_goal.copy()

        # Compute action mask for reachable pixels
        observation["action_mask"] = self._compute_action_mask(depth)

        return observation

    def _compute_action_mask(self, depth):
        """
        Compute action mask for reachable pixel locations.

        Args:
            depth: Depth image (H, W)

        Returns:
            Boolean mask of shape (H*W,) where True indicates reachable pixels
        """
        mask = np.zeros(self.IMAGE_HEIGHT * self.IMAGE_WIDTH, dtype=bool)

        x_min, x_max = self.workspace_bounds["x"]
        y_min, y_max = self.workspace_bounds["y"]

        for y in range(self.IMAGE_HEIGHT):
            for x in range(self.IMAGE_WIDTH):
                pixel_index = y * self.IMAGE_WIDTH + x
                depth_value = depth[y, x]

                # Skip invalid depth values
                if depth_value <= 0 or depth_value > 2.0:
                    continue

                # Convert pixel to world coordinates
                try:
                    coordinates = self.controller.pixel_2_world(
                        pixel_x=x,
                        pixel_y=y,
                        depth=depth_value,
                        height=self.IMAGE_HEIGHT,
                        width=self.IMAGE_WIDTH,
                    )

                    # Check if within workspace bounds (x, y only, z doesn't matter)
                    if x_min <= coordinates[0] <= x_max and y_min <= coordinates[1] <= y_max:
                        mask[pixel_index] = True
                except Exception:
                    # If conversion fails, mark as unreachable
                    continue

        return mask

    # @debug
    def reset_model(self, show_obs=True):
        """
        Method to perform additional reset steps and return an observation.
        Gets called in the parent classes reset method.
        """

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self.controller.actuated_joint_ids] = [
            0,
            -1.57,
            1.57,
            -1.57,
            -1.57,
            0.0,
            0.3,
        ]

        self.object_joint_names = self._gather_object_joint_names()
        try:
            self.ee_body_id = self.controller.body_name2id("ee_link")
        except ValueError:
            self.ee_body_id = None

        # Place objects with minimum spacing to avoid overlap
        min_spacing = 0.08  # Minimum distance between object centers (in meters)
        placed_positions = []

        for joint_name in self.object_joint_names:
            start, end = self.controller.get_joint_qpos_addr(joint_name)

            # Try to find a non-overlapping position
            max_attempts = 50
            for _ in range(max_attempts):
                x = np.random.uniform(low=-0.25, high=0.25)
                y = np.random.uniform(low=-0.65, high=-0.45)

                # Check distance to all previously placed objects
                valid_position = True
                for px, py in placed_positions:
                    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                    if dist < min_spacing:
                        valid_position = False
                        break

                if valid_position:
                    break

            # Use the position (even if not ideal after max_attempts)
            qpos[start] = x
            qpos[start + 1] = y
            placed_positions.append((x, y))

            qpos[start + 2] = np.random.uniform(low=1.0, high=1.5)
            qpos[start + 3: end] = Quaternion.random().unit.elements

        goal_ok = self._set_new_goal(qpos)
        self.unreachable_goal = not goal_ok

        self.set_state(qpos, qvel)

        self.controller.set_group_joint_target(
            group="All", target=qpos[self.controller.actuated_joint_ids]
        )

        # Wait for objects to settle before starting episode
        self.controller.stay(2000, render=self.render_enabled)
        if self.demo_mode:
            self.controller.stay(3000, render=self.render_enabled)
        self.last_grasped_object_pose = None
        self.last_grasped_object_name = None
        self.last_achieved_goal = self._get_end_effector_position()
        # return an observation image
        return self.get_observation(show=self.show_observations)

    def _set_new_goal(self, qpos):
        if not self.object_joint_names:
            self.goal_joint_name = None
            self.desired_goal = np.zeros(3, dtype=np.float32)
            self.last_achieved_goal = np.zeros(3, dtype=np.float32)
            return False

        # First, find all valid candidates (in workspace and with clearance)
        valid_candidates = []
        for joint_name in self.object_joint_names:
            candidate_goal = self._goal_from_qpos(joint_name, qpos)
            if self._goal_in_workspace(candidate_goal) and self._object_has_clearance(
                joint_name, qpos
            ):
                valid_candidates.append((joint_name, candidate_goal))

        if not valid_candidates:
            # No objects have sufficient clearance - truncate episode
            print(
                colored(
                    "[Workspace] No objects with sufficient clearance for grasping",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            self.goal_joint_name = None
            self.desired_goal = np.zeros(3, dtype=np.float32)
            self.target_body_id = None
            self.last_achieved_goal = np.array(
                [0.0, -0.6, self.TABLE_HEIGHT], dtype=np.float32
            )
            return False

        # Randomly select from valid candidates
        chosen_name, chosen_goal = random.choice(valid_candidates)

        self.goal_joint_name = chosen_name
        self.desired_goal = chosen_goal
        self.target_body_id = self.joint_to_body.get(self.goal_joint_name)
        # Default achieved goal above table center
        self.last_achieved_goal = np.array(
            [0.0, -0.6, self.TABLE_HEIGHT], dtype=np.float32
        )
        return True

    def _goal_from_qpos(self, joint_name, qpos):
        start, _ = self.controller.get_joint_qpos_addr(joint_name)
        return np.array(
            [qpos[start], qpos[start + 1], qpos[start + 2]],
            dtype=np.float32,
        )

    def _goal_in_workspace(self, goal):
        x_min, x_max = self.workspace_bounds["x"]
        y_min, y_max = self.workspace_bounds["y"]
        return x_min <= goal[0] <= x_max and y_min <= goal[1] <= y_max

    def _object_has_clearance(self, joint_name, qpos, min_clearance=0.10):
        """
        Check if an object has enough clearance from other objects for grasping.

        Args:
            joint_name: The joint name of the object to check.
            qpos: Current qpos array.
            min_clearance: Minimum distance (in meters) from other objects.
                           Default 0.10 allows for gripper width + object size + margin.

        Returns:
            bool: True if the object has sufficient clearance.
        """
        obj_pos = self._goal_from_qpos(joint_name, qpos)

        for other_name in self.object_joint_names:
            if other_name == joint_name:
                continue
            other_pos = self._goal_from_qpos(other_name, qpos)
            # Check 2D distance (x, y) since z doesn't matter for gripper clearance
            dist = np.linalg.norm(obj_pos[:2] - other_pos[:2])
            if dist < min_clearance:
                return False
        return True

    def compute_reward(self, achieved_goal, desired_goal, grasp_success=False):
        """
        Compute reward based on distance between achieved and desired goal.
        Only gives full reward if position is close AND something was grasped.
        This prevents rewarding empty gripper movements to goal locations.
        No step penalty needed - GAMMA discounting already incentivizes faster grasps.

        Args:
            achieved_goal: The achieved goal position
            desired_goal: The desired goal position
            grasp_success: Whether something was actually grasped (default: False)
        """
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        # Full reward only if position is close AND something was grasped
        if distance <= self.goal_tolerance and grasp_success:
            return 0.5
        clipped_distance = min(distance, 1.0)
        shaping = -0.25 * clipped_distance
        return shaping

    def _cache_object_metadata(self):
        if not hasattr(self, "model"):
            return
        self.object_joint_names = self._gather_object_joint_names()
        self.joint_to_body = {}
        for joint_name in self.object_joint_names:
            try:
                joint_id = self.controller.joint_name2id(joint_name)
            except ValueError:
                continue
            body_id = self.model.jnt_bodyid[joint_id]
            self.joint_to_body[joint_name] = body_id

    def _gather_object_joint_names(self):
        joint_names = []
        for i in range(self.model.njnt):
            name = self.controller.joint_id2name(i)
            if name.startswith("free_joint_"):
                joint_names.append(name)
        joint_names.sort(key=lambda name: self._joint_index(name))
        return joint_names

    @staticmethod
    def _joint_index(name):
        try:
            return int(name.split("_")[-1])
        except (ValueError, IndexError):
            return name

    def _refresh_desired_goal(self):
        if self.goal_joint_name is None:
            self.desired_goal = np.zeros(3, dtype=np.float32)
            self.target_body_id = None
            return
        self.target_body_id = self.joint_to_body.get(self.goal_joint_name)
        if self.target_body_id is None:
            self.desired_goal = np.zeros(3, dtype=np.float32)
            return
        self.desired_goal = self._goal_from_qpos(self.goal_joint_name, self.data.qpos)

    def _get_end_effector_position(self):
        if self.ee_body_id is None:
            self.ee_body_id = self.controller.body_name2id("ee_link")
        return np.array(self.sim.data.xpos[self.ee_body_id], dtype=np.float32)

    def _detect_object_close_to_gripper(self, distance_threshold=0.05):
        if not self.object_joint_names:
            return None, None
        ee_pos = self._get_end_effector_position()
        closest_name = None
        closest_pos = None
        min_dist = float("inf")
        for joint_name in self.object_joint_names:
            pos = self._goal_from_qpos(joint_name, self.data.qpos)
            dist = np.linalg.norm(pos[:2] - ee_pos[:2])
            if dist < distance_threshold and dist < min_dist:
                closest_name = joint_name
                closest_pos = pos.copy()
                min_dist = dist
        return closest_name, closest_pos

    def _capture_grasp_snapshot(self):
        name, pos = self._detect_object_close_to_gripper(distance_threshold=0.08)
        self.last_grasped_object_name = name
        self.last_grasped_object_pose = pos.copy() if pos is not None else None

    def close(self):
        if hasattr(self.controller, "viewer") and self.controller.viewer is not None:
            try:
                self.controller.viewer.close()
            except Exception:
                pass
        try:
            cv.destroyAllWindows()
        except cv.error:
            # Headless mode - destroyAllWindows not available
            pass

    def print_info(self):
        print("Model timestep:", self.model.opt.timestep)
        print("Set number of frames skipped: ", self.frame_skip)
        print("dt = timestep * frame_skip: ", self.dt)
        print("Frames per second = 1/dt: ", self.metadata["video.frames_per_second"])
        print("Actionspace: ", self.action_space)
        print("Observation space:", self.observation_space)
