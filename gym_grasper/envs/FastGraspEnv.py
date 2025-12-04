#!/usr/bin/env python3

import sys
import numpy as np

sys.path.insert(0, "../..")
from gym_grasper.envs.GraspingEnv import GraspEnv


class FastGraspEnv(GraspEnv):
    """
    Fast training environment for Stage 1 pretraining.
    No robot motion - only computes pixel-based rewards based on distance to objects.
    """

    def __init__(
        self,
        file="/UR5+gripper/UR5gripper_2_finger_many_objects.xml",
        image_width=200,
        image_height=200,
        show_obs=False,
        demo=False,
        render=False,
        success_radius=0.02,
        shape_scale=5.0,
    ):
        """
        Initialize FastGraspEnv.

        Args:
            file: MuJoCo XML file path
            image_width: Image width
            image_height: Image height
            show_obs: Whether to show observations
            demo: Demo mode
            render: Whether to render
            success_radius: Distance threshold for success (meters)
            shape_scale: Reward shaping coefficient
        """
        super().__init__(
            file=file,
            image_width=image_width,
            image_height=image_height,
            show_obs=show_obs,
            demo=demo,
            render=render,
        )
        self.success_radius = success_radius
        self.shape_scale = shape_scale

    def step(self, action, record_grasps=False, markers=False, action_info="no info"):
        """
        Fast step: compute reward based on pixel distance to nearest object.
        No robot motion - instant evaluation.

        Args:
            action: (pixel_index, rotation) tuple
            record_grasps: Not used in fast mode
            markers: Not used in fast mode
            action_info: Action info string

        Returns:
            observation, reward, done, info
        """
        # Get current observation
        observation = self.get_observation(show=False)

        # Decode action
        # Note: Agent always outputs (pixel_index, rotation) due to action space,
        # but rotation is not used in fast mode (only pixel proximity matters)
        if self.action_space_type == "multidiscrete":
            pixel_index = action[0]
            _ = action[1]  # Rotation - intentionally unused in fast mode
        else:
            pixel_index = action

        # Convert pixel index to coordinates
        x = pixel_index % self.IMAGE_WIDTH
        y = pixel_index // self.IMAGE_WIDTH

        # Get depth at pixel
        depth = observation["depth"][y, x]

        # Convert pixel to world coordinates
        coordinates = self.controller.pixel_2_world(
            pixel_x=x,
            pixel_y=y,
            depth=depth,
            height=self.IMAGE_HEIGHT,
            width=self.IMAGE_WIDTH,
        )

        # Compute distance to nearest object
        min_dist = float("inf")
        if self.object_joint_names:
            for joint_name in self.object_joint_names:
                obj_pos = self._goal_from_qpos(joint_name, self.data.qpos)
                # 2D distance (x, y) since z doesn't matter for pixel selection
                dist = np.linalg.norm(coordinates[:2] - obj_pos[:2])
                min_dist = min(min_dist, dist)

        # Compute reward
        success = float(min_dist < self.success_radius)
        reward = success - self.shape_scale * min(min_dist, 1.0)

        # Contextual bandit: always done after one action
        done = True

        # Update achieved goal (for compatibility)
        self.last_achieved_goal = coordinates.copy()

        # Info dict (consistent with parent class structure)
        # Note: grasp_success is always 0.0 in fast mode since no actual grasping occurs
        info = {
            "desired_goal": self.desired_goal.copy(),
            "achieved_goal": coordinates.copy(),
            "is_success": success,  # Distance-based success (no actual grasping in fast mode)
            "grasp_success": 0.0,  # Always 0.0 since no robot motion/grasping in fast mode
            "distance": min_dist,
            "her_reward": reward,  # For compatibility (HER not used in fast mode)
        }

        # Ensure observation has action_mask (inherited from parent's get_observation)
        if "action_mask" not in observation:
            observation["action_mask"] = np.ones(
                self.IMAGE_HEIGHT * self.IMAGE_WIDTH, dtype=bool
            )

        return observation, reward, done, info
