"""
Grasping environment implementation.
"""

import numpy as np
import mujoco
import gym
from typing import Dict, Any, Tuple
from .base_env import BaseEnvironment


class GraspingEnvironment(BaseEnvironment):
    """
    Environment for robotic grasping tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the grasping environment.

        Args:
            config: Environment configuration
        """
        super().__init__(config)

        # Environment-specific parameters
        self.success_reward = config['reward']['success_reward']
        self.failure_penalty = config['reward']['failure_penalty']
        self.step_penalty = config['reward']['step_penalty']
        self.distance_weight = config['reward']['distance_reward_weight']

        # Object and gripper settings
        self.object_types = config['objects']['types']
        self.spawn_range = config['objects']['spawn_range']
        self.gripper_threshold = config['gripper']['grasp_threshold']

        # State tracking
        self.target_object_id = None
        self.initial_object_pos = None
        self.grasped = False

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space (joint positions, velocities, end-effector pose, object pose)
        obs_dim = (
            self.model.nq +  # joint positions
            self.model.nv +  # joint velocities
            7 +  # end-effector pose (position + quaternion)
            7   # object pose
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space (discrete: 8 directions + gripper)
        if self.config['action']['type'] == 'discrete':
            self.action_space = gym.spaces.Discrete(self.config['action']['num_actions'])
        else:
            # Continuous action space
            action_dim = self.model.nu
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []

        # Joint positions and velocities
        obs.extend(self.data.qpos[:self.model.nq])
        obs.extend(self.data.qvel[:self.model.nv])

        # End-effector pose
        ee_pos, ee_quat = self._get_end_effector_pose()
        obs.extend(ee_pos)
        obs.extend(ee_quat)

        # Object pose
        if self.target_object_id is not None:
            obj_pos, obj_quat = self._get_object_pose()
            obs.extend(obj_pos)
            obs.extend(obj_quat)
        else:
            obs.extend([0.0] * 7)  # Zero pose if no object

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action."""
        reward = 0.0

        # Step penalty
        reward += self.step_penalty

        # Success reward
        if self._check_success():
            reward += self.success_reward
            return reward

        # Distance-based reward
        if self.target_object_id is not None:
            ee_pos, _ = self._get_end_effector_pose()
            obj_pos, _ = self._get_object_pose()
            distance = np.linalg.norm(ee_pos - obj_pos)
            reward += self.distance_weight * (1.0 / (1.0 + distance))

        # Contact reward
        if self._check_contact():
            reward += 0.1

        return reward

    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Episode length limit
        if self.episode_step >= self.max_episode_steps:
            return True

        # Success condition
        if self._check_success():
            return True

        # Failure condition (object fell off table)
        if self.target_object_id is not None:
            obj_pos, _ = self._get_object_pose()
            if obj_pos[2] < -0.1:  # Below table
                return True

        return False

    def _apply_action(self, action: np.ndarray):
        """Apply action to the environment."""
        if self.config['action']['type'] == 'discrete':
            self._apply_discrete_action(action)
        else:
            self._apply_continuous_action(action)

    def _apply_discrete_action(self, action: int):
        """Apply discrete action."""
        # Define action mappings
        action_map = {
            0: [0.1, 0, 0, 0],    # Move right
            1: [-0.1, 0, 0, 0],   # Move left
            2: [0, 0.1, 0, 0],    # Move forward
            3: [0, -0.1, 0, 0],   # Move backward
            4: [0, 0, 0.1, 0],    # Move up
            5: [0, 0, -0.1, 0],   # Move down
            6: [0, 0, 0, 0.1],    # Close gripper
            7: [0, 0, 0, -0.1],   # Open gripper
        }

        if action in action_map:
            self.data.ctrl[:] = action_map[action]

    def _apply_continuous_action(self, action: np.ndarray):
        """Apply continuous action."""
        # Scale actions
        action_scale = self.config['action']['action_scale']
        self.data.ctrl[:] = action * action_scale

    def _reset_environment(self):
        """Reset environment-specific state."""
        # Reset object position
        self._spawn_object()

        # Reset gripper
        self.grasped = False

        # Reset robot to home position
        self._reset_robot_pose()

    def _spawn_object(self):
        """Spawn a random object in the scene."""
        # Random object type
        obj_type = np.random.choice(self.object_types)

        # Random position within spawn range
        x = np.random.uniform(self.spawn_range['x'][0], self.spawn_range['x'][1])
        y = np.random.uniform(self.spawn_range['y'][0], self.spawn_range['y'][1])
        z = np.random.uniform(self.spawn_range['z'][0], self.spawn_range['z'][1])

        # Set object position
        if self.target_object_id is not None:
            self.data.qpos[self.target_object_id:self.target_object_id+3] = [x, y, z]

        self.initial_object_pos = np.array([x, y, z])

    def _reset_robot_pose(self):
        """Reset robot to home position."""
        # Set joint positions to zero
        self.data.qpos[:self.model.nq] = 0
        mujoco.mj_forward(self.model, self.data)

    def _get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation."""
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        pos = self.data.xpos[ee_id].copy()
        quat = self.data.xquat[ee_id].copy()
        return pos, quat

    def _get_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get object position and orientation."""
        if self.target_object_id is not None:
            pos = self.data.qpos[self.target_object_id:self.target_object_id+3].copy()
            quat = self.data.qpos[self.target_object_id+3:self.target_object_id+7].copy()
            return pos, quat
        return np.zeros(3), np.array([1, 0, 0, 0])

    def _check_contact(self) -> bool:
        """Check if gripper is in contact with object."""
        contacts = self._get_contact_forces()
        for contact in contacts:
            if contact['dist'] < 0.01:  # Contact threshold
                return True
        return False

    def _check_success(self) -> bool:
        """Check if grasping was successful."""
        if self.target_object_id is None:
            return False

        # Check if object is lifted
        obj_pos, _ = self._get_object_pose()
        initial_z = self.initial_object_pos[2]

        # Success if object is lifted above initial position
        if obj_pos[2] > initial_z + 0.05:
            return True

        return False
