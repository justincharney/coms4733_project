"""
Environment utilities for robotic grasping environments.
"""

import numpy as np
import mujoco
from typing import Dict, Any, List
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_joint_positions(model: mujoco.MjModel, data: mujoco.MjData,
                        joint_names: List[str]) -> np.ndarray:
    """
    Get joint positions for specified joints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names

    Returns:
        Array of joint positions
    """
    positions = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        positions.append(data.qpos[joint_id])
    return np.array(positions)


def get_joint_velocities(model: mujoco.MjModel, data: mujoco.MjData,
                         joint_names: List[str]) -> np.ndarray:
    """
    Get joint velocities for specified joints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names

    Returns:
        Array of joint velocities
    """
    velocities = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        velocities.append(data.qvel[joint_id])
    return np.array(velocities)


def get_body_position(model: mujoco.MjModel, data: mujoco.MjData,
                      body_name: str) -> np.ndarray:
    """
    Get body position.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body

    Returns:
        Body position (x, y, z)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def get_body_quaternion(model: mujoco.MjModel, data: mujoco.MjData,
                        body_name: str) -> np.ndarray:
    """
    Get body quaternion.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of body

    Returns:
        Body quaternion (w, x, y, z)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xquat[body_id].copy()


def get_gripper_state(model: mujoco.MjModel, data: mujoco.MjData,
                      gripper_joints: List[str]) -> np.ndarray:
    """
    Get gripper state (open/close).

    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_joints: List of gripper joint names

    Returns:
        Gripper state
    """
    positions = get_joint_positions(model, data, gripper_joints)
    return positions


def set_joint_positions(model: mujoco.MjModel, data: mujoco.MjData,
                        joint_names: List[str], positions: np.ndarray) -> None:
    """
    Set joint positions.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names
        positions: Target positions
    """
    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[joint_id] = positions[i]


def set_joint_velocities(model: mujoco.MjModel, data: mujoco.MjData,
                         joint_names: List[str], velocities: np.ndarray) -> None:
    """
    Set joint velocities.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names
        velocities: Target velocities
    """
    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qvel[joint_id] = velocities[i]


def check_collision(model: mujoco.MjModel, data: mujoco.MjData,
                    body1_name: str, body2_name: str) -> bool:
    """
    Check if two bodies are in collision.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body1_name: Name of first body
        body2_name: Name of second body

    Returns:
        True if bodies are in collision
    """
    # Get body IDs
    body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

    # Check collision
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == body1_id and contact.geom2 == body2_id) or \
           (contact.geom1 == body2_id and contact.geom2 == body1_id):
            return True
    return False


def get_distance_between_bodies(model: mujoco.MjModel, data: mujoco.MjData,
                                body1_name: str, body2_name: str) -> float:
    """
    Get distance between two bodies.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body1_name: Name of first body
        body2_name: Name of second body

    Returns:
        Distance between bodies
    """
    pos1 = get_body_position(model, data, body1_name)
    pos2 = get_body_position(model, data, body2_name)
    return np.linalg.norm(pos1 - pos2)


def is_grasping(model: mujoco.MjModel, data: mujoco.MjData,
                gripper_body: str, object_body: str,
                grasp_threshold: float = 0.05) -> bool:
    """
    Check if gripper is grasping an object.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_body: Name of gripper body
        object_body: Name of object body
        grasp_threshold: Distance threshold for grasping

    Returns:
        True if grasping
    """
    distance = get_distance_between_bodies(model, data, gripper_body, object_body)
    return distance < grasp_threshold


def is_object_lifted(model: mujoco.MjModel, data: mujoco.MjData,
                     object_body: str, table_height: float = 0.0,
                     lift_threshold: float = 0.1) -> bool:
    """
    Check if object is lifted above table.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_body: Name of object body
        table_height: Height of table
        lift_threshold: Height threshold for lifting

    Returns:
        True if object is lifted
    """
    pos = get_body_position(model, data, object_body)
    return pos[2] > table_height + lift_threshold


def reset_robot_to_home(model: mujoco.MjModel, data: mujoco.MjData,
                        joint_names: List[str], home_positions: np.ndarray) -> None:
    """
    Reset robot to home position.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names
        home_positions: Home positions for joints
    """
    set_joint_positions(model, data, joint_names, home_positions)
    set_joint_velocities(model, data, joint_names, np.zeros(len(joint_names)))
    mujoco.mj_forward(model, data)


def spawn_object_randomly(model: mujoco.MjModel, data: mujoco.MjData,
                          object_body: str, spawn_radius: float = 0.3,
                          table_height: float = 0.0) -> None:
    """
    Spawn object at random position on table.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_body: Name of object body
        spawn_radius: Radius for random spawning
        table_height: Height of table
    """
    # Generate random position
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, spawn_radius)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = table_height + 0.1  # Slightly above table

    # Set object position
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    data.qpos[body_id] = [x, y, z, 1, 0, 0, 0]  # Position and quaternion

    # Reset velocities
    data.qvel[body_id] = 0

    mujoco.mj_forward(model, data)


def compute_reward(grasped: bool, lifted: bool, collision: bool,
                   time_penalty: float = 0.01) -> float:
    """
    Compute reward based on task progress.

    Args:
        grasped: Whether object is grasped
        lifted: Whether object is lifted
        collision: Whether there was a collision
        time_penalty: Penalty for time step

    Returns:
        Reward value
    """
    reward = 0.0

    if grasped:
        reward += 10.0
    if lifted:
        reward += 5.0
    if collision:
        reward -= 1.0

    reward -= time_penalty

    return reward


def normalize_observation(obs: np.ndarray, obs_mean: np.ndarray,
                          obs_std: np.ndarray) -> np.ndarray:
    """
    Normalize observation using mean and standard deviation.

    Args:
        obs: Observation to normalize
        obs_mean: Mean of observations
        obs_std: Standard deviation of observations

    Returns:
        Normalized observation
    """
    return (obs - obs_mean) / (obs_std + 1e-8)


def denormalize_observation(obs: np.ndarray, obs_mean: np.ndarray,
                            obs_std: np.ndarray) -> np.ndarray:
    """
    Denormalize observation using mean and standard deviation.

    Args:
        obs: Normalized observation
        obs_mean: Mean of observations
        obs_std: Standard deviation of observations

    Returns:
        Denormalized observation
    """
    return obs * obs_std + obs_mean


def clip_action(action: np.ndarray, action_low: np.ndarray,
                action_high: np.ndarray) -> np.ndarray:
    """
    Clip action to valid range.

    Args:
        action: Action to clip
        action_low: Lower bound for actions
        action_high: Upper bound for actions

    Returns:
        Clipped action
    """
    return np.clip(action, action_low, action_high)


def compute_observation_space_size(config: Dict[str, Any]) -> int:
    """
    Compute observation space size from configuration.

    Args:
        config: Environment configuration

    Returns:
        Size of observation space
    """
    size = 0

    if config.get('include_joint_positions', False):
        size += 7  # 6 DOF arm + 1 gripper

    if config.get('include_joint_velocities', False):
        size += 7  # 6 DOF arm + 1 gripper

    if config.get('include_gripper_state', False):
        size += 1

    if config.get('include_object_positions', False):
        size += 3  # x, y, z

    if config.get('include_robot_pose', False):
        size += 7  # position (3) + quaternion (4)

    return size


def compute_action_space_size(config: Dict[str, Any]) -> int:
    """
    Compute action space size from configuration.

    Args:
        config: Environment configuration

    Returns:
        Size of action space
    """
    return config.get('action_dim', 7)  # 6 DOF arm + 1 gripper
