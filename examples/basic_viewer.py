"""
Basic MuJoCo viewer example.
Load and visualize a robot model with interactive controls.
"""

import mujoco
import mujoco.viewer
import numpy as np


def main():
    # Load the model (change path as needed)
    model_path = "../models/robot_arm.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"Model loaded: {model.nq} DOF")
    print(f"Actuators: {model.nu}")
    print(f"Sensors: {model.nsensor}")

    # Launch the passive viewer
    # Use arrow keys, mouse, and space bar to interact
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            # Step the simulation
            step_start = data.time

            # Apply zero control (or add your controller here)
            data.ctrl[:] = 0

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer at 60 fps
            viewer.sync()

            # Optional: slow down to real-time
            # time.sleep(max(0, model.opt.timestep - (data.time - step_start)))


if __name__ == "__main__":
    main()
