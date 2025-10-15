"""
Utility functions for MuJoCo robotics simulations.
"""

import mujoco
import numpy as np
from typing import Tuple, Optional


def get_joint_limits(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get joint limits from the model.

    Args:
        model: MuJoCo model

    Returns:
        Tuple of (lower_limits, upper_limits)
    """
    lower = model.jnt_range[:, 0]
    upper = model.jnt_range[:, 1]
    return lower, upper


def set_joint_angles(model: mujoco.MjModel, data: mujoco.MjData,
                     angles: np.ndarray, joint_names: Optional[list] = None):
    """
    Set joint angles by name or by index.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        angles: Joint angles to set
        joint_names: Optional list of joint names (if None, sets by index)
    """
    if joint_names is not None:
        for name, angle in zip(joint_names, angles):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx = model.jnt_qposadr[joint_id]
            data.qpos[qpos_idx] = angle
    else:
        data.qpos[:len(angles)] = angles

    mujoco.mj_forward(model, data)


def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData,
                  body_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get position and orientation of a body.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body

    Returns:
        Tuple of (position, quaternion)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    position = data.xpos[body_id].copy()
    quaternion = data.xquat[body_id].copy()
    return position, quaternion


def get_jacobian(model: mujoco.MjModel, data: mujoco.MjData,
                 body_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobian for a body.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body

    Returns:
        Tuple of (translational_jacobian, rotational_jacobian)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    return jacp, jacr


def compute_inverse_kinematics(model: mujoco.MjModel, data: mujoco.MjData,
                               body_name: str, target_pos: np.ndarray,
                               max_iterations: int = 100, tolerance: float = 1e-3) -> bool:
    """
    Simple IK solver using Jacobian pseudo-inverse.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the end-effector body
        target_pos: Target position [x, y, z]
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        True if converged, False otherwise
    """
    for _ in range(max_iterations):
        # Get current position
        current_pos, _ = get_body_pose(model, data, body_name)

        # Compute error
        error = target_pos - current_pos

        # Check convergence
        if np.linalg.norm(error) < tolerance:
            return True

        # Get Jacobian
        jacp, _ = get_jacobian(model, data, body_name)

        # Compute pseudo-inverse
        jacp_pinv = np.linalg.pinv(jacp)

        # Compute joint velocity
        dq = jacp_pinv @ error * 0.5  # Scale factor for stability

        # Update joint positions
        data.qpos[:model.nv] += dq

        # Forward kinematics
        mujoco.mj_forward(model, data)

    return False


def apply_wrench(model: mujoco.MjModel, data: mujoco.MjData,
                 body_name: str, force: np.ndarray, torque: np.ndarray):
    """
    Apply force and torque to a body.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body
        force: Force vector [fx, fy, fz]
        torque: Torque vector [tx, ty, tz]
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[body_id, :3] = force
    data.xfrc_applied[body_id, 3:] = torque


def get_contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """
    Get all contact forces in the simulation.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        List of dictionaries containing contact information
    """
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]

        # Get contact force
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        # Get geometry names
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        contacts.append({
            'geom1': geom1_name,
            'geom2': geom2_name,
            'pos': contact.pos.copy(),
            'force': force.copy(),
            'dist': contact.dist
        })

    return contacts


def print_model_info(model: mujoco.MjModel):
    """
    Print useful information about the model.

    Args:
        model: MuJoCo model
    """
    print("=" * 50)
    print("MuJoCo Model Information")
    print("=" * 50)
    print(f"Model name: {model.names().decode() if model.names() else 'Unnamed'}")
    print(f"DOF (nq): {model.nq}")
    print(f"Velocities (nv): {model.nv}")
    print(f"Actuators (nu): {model.nu}")
    print(f"Sensors: {model.nsensor}")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Geoms: {model.ngeom}")
    print(f"Timestep: {model.opt.timestep}")
    print(f"Gravity: {model.opt.gravity}")

    print("\nJoints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        print(f"  {i}: {joint_name} (type: {joint_type})")

    print("\nActuators:")
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")

    print("=" * 50)
