# MuJoCo Robotics Simulation

A base repository for robotics simulation using MuJoCo physics engine. This repository provides example models, controllers, and utilities to get started with robot simulation quickly.

## Features

- Pre-configured MuJoCo environment
- Example robot models (pendulum, robot arm)
- Basic control examples (PD controller, sinusoidal motion)
- Simulation recording and visualization tools
- Utility functions for kinematics, dynamics, and contacts

## Project Structure

```
mujoco_robotics/
├── models/              # XML model files
│   ├── simple_pendulum.xml
│   └── robot_arm.xml
├── examples/            # Example simulation scripts
│   ├── basic_viewer.py
│   ├── simple_controller.py
│   └── record_simulation.py
├── src/                 # Utility modules
│   ├── __init__.py
│   └── utils.py
├── assets/              # Generated assets (videos, plots)
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to this repository:

```bash
cd mujoco_robotics
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
```

## Quick Start

### 1. Basic Viewer

Launch an interactive viewer to visualize and interact with a robot model:

```bash
cd examples
python basic_viewer.py
```

**Controls:**

- Left mouse button: Rotate view
- Right mouse button: Zoom
- Middle mouse button: Pan
- Space: Pause/resume
- ESC: Exit

### 2. Simple Controller

Run a simulation with basic control:

```bash
python simple_controller.py
```

This demonstrates:

- Sinusoidal motion control
- PD (Proportional-Derivative) controller
- Real-time visualization

### 3. Record Simulation

Record simulation data and render videos:

```bash
python record_simulation.py
```

Outputs:

- `../assets/simulation_results.png` - Joint trajectory plots
- `../assets/simulation.mp4` - Rendered video (requires imageio-ffmpeg)

## Creating Custom Models

MuJoCo uses XML files to define robots and environments. Here's a minimal example:

```xml
<mujoco model="my_robot">
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom name="geom1" type="cylinder" size="0.05 0.3"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" gear="1"/>
  </actuator>
</mujoco>
```

Save as `models/my_robot.xml` and load with:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("models/my_robot.xml")
```

## Using Utility Functions

The `src/utils.py` module provides helpful functions:

```python
from src.utils import *

# Print model information
print_model_info(model)

# Get end-effector position
pos, quat = get_body_pose(model, data, "end_effector")

# Compute Jacobian
jacp, jacr = get_jacobian(model, data, "end_effector")

# Simple IK solver
target = np.array([0.3, 0.3, 0.5])
success = compute_inverse_kinematics(model, data, "end_effector", target)

# Get contact forces
contacts = get_contact_forces(model, data)
```

## Writing Controllers

Example PD controller:

```python
def pd_controller(target_pos, current_pos, current_vel, kp=10.0, kd=2.0):
    """PD control law"""
    error = target_pos - current_pos
    ctrl = kp * error - kd * current_vel
    return ctrl

# In simulation loop:
data.ctrl[:] = pd_controller(target, data.qpos, data.qvel)
mujoco.mj_step(model, data)
```

## Advanced Topics

### Rendering Offscreen

```python
import mujoco

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
pixels = renderer.render()
```

### Sensor Data

```python
# Access sensor readings
sensor_data = data.sensordata
print(f"Sensor values: {sensor_data}")
```

### Contact Handling

```python
for i in range(data.ncon):
    contact = data.contact[i]
    print(f"Contact {i}: distance={contact.dist}")
```
