"""
Example showing how to record simulation data and render videos.
Demonstrates offline rendering and data logging.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_and_record(model, data, duration=5.0, render=True):
    """
    Run simulation and record state data.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        duration: Simulation duration in seconds
        render: Whether to render frames

    Returns:
        Dictionary containing recorded data
    """
    # Calculate number of steps
    n_steps = int(duration / model.opt.timestep)

    # Initialize data storage
    recorded_data = {
        'time': np.zeros(n_steps),
        'qpos': np.zeros((n_steps, model.nq)),
        'qvel': np.zeros((n_steps, model.nv)),
        'ctrl': np.zeros((n_steps, model.nu)),
    }

    # Initialize renderer if needed
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []

    print(f"Simulating for {duration} seconds...")

    # Simulation loop
    for step in range(n_steps):
        # Simple sinusoidal control
        t = data.time
        data.ctrl[:] = 2.0 * np.sin(2 * np.pi * 0.5 * t)

        # Record data
        recorded_data['time'][step] = data.time
        recorded_data['qpos'][step] = data.qpos.copy()
        recorded_data['qvel'][step] = data.qvel.copy()
        recorded_data['ctrl'][step] = data.ctrl.copy()

        # Render frame
        if render and step % 2 == 0:  # Record every other frame
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels.copy())

        # Step simulation
        mujoco.mj_step(model, data)

        # Print progress
        if step % 500 == 0:
            print(f"Step {step}/{n_steps}")

    print("Simulation complete!")

    if render:
        recorded_data['frames'] = frames

    return recorded_data


def plot_results(recorded_data, save_path='../assets/simulation_results.png'):
    """
    Plot simulation results.

    Args:
        recorded_data: Dictionary with recorded simulation data
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    time = recorded_data['time']

    # Plot positions
    axes[0].plot(time, recorded_data['qpos'])
    axes[0].set_ylabel('Joint Positions (rad)')
    axes[0].set_title('Joint Positions over Time')
    axes[0].grid(True)
    axes[0].legend([f'Joint {i}' for i in range(recorded_data['qpos'].shape[1])])

    # Plot velocities
    axes[1].plot(time, recorded_data['qvel'])
    axes[1].set_ylabel('Joint Velocities (rad/s)')
    axes[1].set_title('Joint Velocities over Time')
    axes[1].grid(True)

    # Plot control signals
    axes[2].plot(time, recorded_data['ctrl'])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Control Signal')
    axes[2].set_title('Control Signals over Time')
    axes[2].grid(True)

    plt.tight_layout()

    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

    plt.show()


def save_video(frames, save_path='../assets/simulation.mp4', fps=30):
    """
    Save rendered frames as video.

    Args:
        frames: List of frame arrays
        save_path: Path to save video
        fps: Frames per second
    """
    try:
        import imageio
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Video saved to {save_path}")
    except ImportError:
        print("imageio not installed. Install with: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"Error saving video: {e}")


def main():
    # Load model
    model_path = "../models/simple_pendulum.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set initial state
    data.qpos[:] = np.pi / 4  # Start at 45 degrees
    mujoco.mj_forward(model, data)

    # Run simulation and record
    recorded_data = simulate_and_record(model, data, duration=5.0, render=True)

    # Plot results
    plot_results(recorded_data)

    # Save video if frames were recorded
    if 'frames' in recorded_data:
        save_video(recorded_data['frames'])


if __name__ == "__main__":
    main()
