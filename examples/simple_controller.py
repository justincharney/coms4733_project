"""
Simple controller example with sinusoidal motion.
Demonstrates basic actuator control in MuJoCo.
"""

import mujoco
import mujoco.viewer
import numpy as np


def sinusoidal_controller(model, data, t):
    """
    Simple sinusoidal controller for demonstration.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        t: Current simulation time

    Returns:
        Control signals for actuators
    """
    ctrl = np.zeros(model.nu)

    # Apply sinusoidal control to each actuator
    for i in range(model.nu):
        amplitude = 2.0  # Control amplitude
        frequency = 0.5 + i * 0.3  # Varying frequencies for different joints
        ctrl[i] = amplitude * np.sin(2 * np.pi * frequency * t)

    return ctrl


def pd_controller(model, data, target_pos, kp=10.0, kd=2.0):
    """
    Simple PD controller for joint positions.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: Target joint positions
        kp: Proportional gain
        kd: Derivative gain

    Returns:
        Control signals
    """
    # Get current joint positions and velocities
    qpos = data.qpos[:model.nu]
    qvel = data.qvel[:model.nu]

    # Compute PD control
    ctrl = kp * (target_pos - qpos) - kd * qvel

    return ctrl


def main():
    # Load the model
    model_path = "../models/robot_arm.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"Simulating {model.nq} DOF robot")
    print("Press ESC to exit")

    # Set initial configuration
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    # Choose controller type
    controller_type = "sinusoidal"  # Options: "sinusoidal", "pd"

    # Target positions for PD controller
    target_positions = np.array([np.pi/4, -np.pi/3, 0, 0])[:model.nu]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Choose and apply controller
            if controller_type == "sinusoidal":
                data.ctrl[:] = sinusoidal_controller(model, data, data.time)
            elif controller_type == "pd":
                data.ctrl[:] = pd_controller(model, data, target_positions)
            else:
                data.ctrl[:] = 0

            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()


if __name__ == "__main__":
    main()
