"""
Base environment class for robotic grasping tasks.
"""

import gym
import mujoco
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseEnvironment(gym.Env, ABC):
    """
    Abstract base class for robotic grasping environments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base environment.

        Args:
            config: Environment configuration dictionary
        """
        super().__init__()
        self.config = config
        self.model = None
        self.data = None
        self.episode_step = 0
        self.max_episode_steps = config.get('max_episode_steps', 1000)

        # Initialize MuJoCo
        self._load_model()
        self._setup_spaces()

    def _load_model(self):
        """Load the MuJoCo model."""
        model_path = self.config['model_path']
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    @abstractmethod
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        pass

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        pass

    @abstractmethod
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for the current state and action."""
        pass

    @abstractmethod
    def _is_done(self) -> bool:
        """Check if episode is done."""
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            observation, reward, done, info
        """
        # Apply action
        self._apply_action(action)

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation and reward
        observation = self._get_observation()
        reward = self._compute_reward(action)
        done = self._is_done()

        # Update episode step
        self.episode_step += 1

        # Create info dictionary
        info = {
            'episode_step': self.episode_step,
            'success': self._check_success(),
            'contact_forces': self._get_contact_forces()
        }

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)

        # Reset episode step
        self.episode_step = 0

        # Reset environment-specific state
        self._reset_environment()

        # Get initial observation
        observation = self._get_observation()

        return observation

    @abstractmethod
    def _apply_action(self, action: np.ndarray):
        """Apply action to the environment."""
        pass

    @abstractmethod
    def _reset_environment(self):
        """Reset environment-specific state."""
        pass

    @abstractmethod
    def _check_success(self) -> bool:
        """Check if task was successful."""
        pass

    def _get_contact_forces(self) -> list:
        """Get contact forces in the simulation."""
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)

            contacts.append({
                'geom1': contact.geom1,
                'geom2': contact.geom2,
                'pos': contact.pos.copy(),
                'force': force.copy(),
                'dist': contact.dist
            })
        return contacts

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            # Use MuJoCo viewer
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.sync()
        elif mode == 'rgb_array':
            # Return RGB array
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        """Close the environment."""
        pass
