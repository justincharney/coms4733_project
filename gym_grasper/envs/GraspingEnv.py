#!/usr/bin/env python3

# Author: Paul Daniel (pdd@mp.aau.dk)
import sys

sys.path.insert(0, "..")
import copy
import math
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import gym
import mujoco as mj
import numpy as np
from gym import spaces, utils
from pyquaternion import Quaternion
from termcolor import colored

from decorators import *
from gym_grasper.controller.MujocoController import (
    DummyViewer,
    MJ_Controller,
    MjSimWrapper,
)


class GraspEnv(gym.Env, utils.EzPickle):
    # def __init__(self, file='/UR5+gripper/UR5gripper_2_finger.xml', image_width=200, image_height=200, show_obs=True, demo=False, render=False):
    # def __init__(self, file='/UR5+gripper/UR5gripper_2_finger.xml', image_width=200, image_height=200, show_obs=True, demo=False, render=False):
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
        self.viewer = DummyViewer() if render else None
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
        # Parent class will step once during init to set up the observation space, controller is not yet available at that time.
        # Therefore we simply return a dictionary of zeros of the appropriate size.
        if not self.initialized:
            # self.current_observation = np.zeros((200,200,4))
            self.current_observation = defaultdict()
            self.current_observation["rgb"] = np.zeros(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
            )
            self.current_observation["depth"] = np.zeros(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            )
            self.current_observation["desired_goal"] = self.desired_goal.copy()
            self.current_observation["achieved_goal"] = self.last_achieved_goal.copy()
            reward = 0
        else:
            if self.step_called == 1:
                self.current_observation = self.get_observation(show=False)
            self._refresh_desired_goal()
            goal_before_action = self.desired_goal.copy()
            achieved_goal = self._get_end_effector_position()
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

            coordinates = self.controller.pixel_2_world(
                pixel_x=x,
                pixel_y=y,
                depth=depth,
                height=self.IMAGE_HEIGHT,
                width=self.IMAGE_WIDTH,
            )
            print(
                colored(
                    "Action ({}): Pixel X: {}, Pixel Y: {}, Rotation: {} ({} deg)".format(
                        action_info, x, y, rotation, self.rotations[rotation]
                    ),
                    color="blue",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    "Transformed into world coordinates: {}".format(coordinates[:2]),
                    color="blue",
                    attrs=["bold"],
                )
            )

            # Check for coordinates we don't need to try to save some time
            if coordinates[2] < 0.8 or coordinates[1] > -0.3:
                print(
                    colored(
                        "Skipping execution due to bad depth value!",
                        color="red",
                        attrs=["bold"],
                    )
                )
                # Binary reward
                reward = 0.0
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
            her_reward = float(self.compute_reward(achieved_goal, goal_before_action))
            reward = her_reward
            if self.initialized:
                print(
                    colored(
                        "Reward received during step: {}".format(reward),
                        color="yellow",
                        attrs=["bold"],
                    )
                )

            self.current_observation = self.get_observation(show=self.show_observations)

        self.step_called += 1
        info["desired_goal"] = goal_before_action.copy()
        info["achieved_goal"] = self.last_achieved_goal.copy()
        info["is_success"] = her_reward
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
        Legacy method, not used in the current setup. May be used to directly set the joint values to a desired position.
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
        # return np.round(max(self.TABLE_HEIGHT, self.TABLE_HEIGHT + height_action * (depth_height - self.TABLE_HEIGHT)/self.action_space.nvec[1]), decimals=3)

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

        # Check result1 for max steps reached => if so it got stuck in a bad position
        if result1[:3] == "max":
            result_rotate = "Skipped"
            steps_rotate = 0
            result2 = "Skipped"
            coordinates_2 = None
            steps2 = 0
            result_grasp = False
            reach_success = False

        else:
            # Rotate gripper according to second action dimension
            result_rotate = self.rotate_wrist_3_joint_to_value(self.rotations[rotation])
            steps_rotate = self.controller.last_steps

            self.controller.open_gripper(
                half=True, render=render, quiet=True, plot=plot
            )

            # Move to grasping height
            coordinates_2 = copy.deepcopy(coordinates)
            coordinates_2[2] = max(self.TABLE_HEIGHT, coordinates_2[2] - 0.01)
            result2 = self.controller.move_ee(
                coordinates_2,
                max_steps=300,
                quiet=True,
                render=render,
                marker=markers,
                tolerance=0.01,
                plot=plot,
            )
            steps2 = self.controller.last_steps

            # If we can't reach the desired grasping position, don't grasp
            if result2[:3] == "max":
                result2 = "Could not reach target location"
                result_grasp = False
                reach_success = False

            else:
                reach_success = True
                self.controller.stay(100, render=render)
                result_grasp = self.controller.grasp(
                    render=render, quiet=True, marker=markers, plot=plot
                )
                if result_grasp:
                    self._capture_grasp_snapshot()

        self.controller.actuators[0][4].Kp = 10.0

        # Move back above center of table
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

        # Move to drop position
        result4 = self.controller.move_ee(
            [0.6, 0.0, 1.15],
            max_steps=1200,
            quiet=True,
            render=render,
            plot=plot,
            marker=markers,
            tolerance=0.01,
        )
        steps4 = self.controller.last_steps

        # self.controller.stay(500)

        result_final = "Skipped"

        if result_grasp:
            if not self.demo_mode:
                # Perform check if object is in gripper
                result_final = self.controller.close_gripper(
                    max_steps=1000, render=render, quiet=True, marker=markers, plot=plot
                )
            else:
                result_final = self.controller.close_gripper(
                    max_steps=100, render=render, quiet=True, marker=markers, plot=plot
                )

        final_str = "Nothing in the gripper"
        if result_final[:3] == "max":
            final_str = "Object in the gripper"

        grasped_something = result_final[:3] == "max" and result_grasp
        if not grasped_something:
            self.last_grasped_object_pose = None
            self.last_grasped_object_name = None

        if grasped_something and record_grasps:
            capture_rgb, depth = self.controller.get_image_data(
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
        result_rotate_back = self.rotate_wrist_3_joint_to_value(0)

        self.controller.actuators[0][4].Kp = 20.0

        # if self.demo_mode:
        #     self.controller.stay(200, render=render)
        #     return 'demo'

        # else:
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
        print(
            f"Move to grasping position (z={np.round(coordinates_2[2], decimals=4) if isinstance(coordinates_2, np.ndarray) else 0}):".ljust(
                40, " "
            ),
            result2,
            ",",
            steps2,
            "steps",
        )
        print("Grasped anything?: ".ljust(40, " "), result_grasp)
        print("Move to center: ".ljust(40, " "), result3, ",", steps3, "steps")
        print("Move to drop position: ".ljust(40, " "), result4, ",", steps4, "steps")
        print("Final finger check: ".ljust(40, " "), final_str)
        print("Open gripper: ".ljust(40, " "), result_open, ",", steps_open, "steps")

        if result1 == result2 == result3 == result4 == result_open == "success":
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

    # @debug
    # @dict2list
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
        observation = defaultdict()
        observation["rgb"] = rgb
        observation["depth"] = depth
        observation["desired_goal"] = self.desired_goal.copy()
        observation["achieved_goal"] = self.last_achieved_goal.copy()

        return observation

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

        for joint_name in self.object_joint_names:
            start, end = self.controller.get_joint_qpos_addr(joint_name)
            qpos[start] = np.random.uniform(low=-0.25, high=0.25)
            qpos[start + 1] = np.random.uniform(low=-0.77, high=-0.43)
            # qpos[start+2] = 1.0
            qpos[start + 2] = np.random.uniform(low=1.0, high=1.5)
            qpos[start + 3 : end] = Quaternion.random().unit.elements
        self._set_new_goal(qpos)

        #########################################################################
        # Reset for IT4, older versions of IT5

        # n_boxes = 3
        # n_balls = 3

        # for j in ['rot', 'x', 'y', 'z']:
        #     for i in range(1,n_boxes+1):
        #         joint_name = 'box_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]

        #     for i in range(1,n_balls+1):
        #         joint_name = 'ball_' + str(i) + '_' + j
        #         q_adr = self.model.get_joint_qpos_addr(joint_name)
        #         if j == 'x':
        #             qpos[q_adr] = np.random.uniform(low=-0.25, high=0.25)
        #         elif j == 'y':
        #             qpos[q_adr] = np.random.uniform(low=-0.17, high=0.17)
        #         elif j == 'z':
        #             qpos[q_adr] = 0.0
        #         elif j == 'rot':
        #             start, end = q_adr
        #             qpos[start:end] = [1., 0., 0., 0.]
        #########################################################################

        self.set_state(qpos, qvel)

        self.controller.set_group_joint_target(
            group="All", target=qpos[self.controller.actuated_joint_ids]
        )

        # Turn this on for training, so the objects drop down before the observation
        self.controller.stay(1000, render=self.render_enabled)
        if self.demo_mode:
            self.controller.stay(5000, render=self.render_enabled)
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
            return

        self.goal_joint_name = random.choice(self.object_joint_names)
        self.desired_goal = self._goal_from_qpos(self.goal_joint_name, qpos)
        self.target_body_id = self.joint_to_body.get(self.goal_joint_name)
        # Default achieved goal above table center
        self.last_achieved_goal = np.array(
            [0.0, -0.6, self.TABLE_HEIGHT], dtype=np.float32
        )

    def _goal_from_qpos(self, joint_name, qpos):
        start, _ = self.controller.get_joint_qpos_addr(joint_name)
        return np.array(
            [qpos[start], qpos[start + 1], qpos[start + 2]],
            dtype=np.float32,
        )

    def compute_reward(self, achieved_goal, desired_goal):
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        return 1.0 if distance <= self.goal_tolerance else 0.0

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
