# Robotic Grasping using Deep Reinforcement Learning

A deep reinforcement learning for robotic grasping tasks using MuJoCo physics simulation.

## Project Overview

This project implements a deep reinforcement learning system for robotic grasping, featuring:
- **Environment**: MuJoCo-based robotic arm simulation with gripper
- **RL Algorithm**: Deep Q-Network (DQN) and Policy Gradient methods
- **Task**: Object grasping and manipulation
- **Visualization**: Real-time training monitoring and evaluation

## Project Structure

```
coms4733_project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config/                            # Configuration files
│   ├── environments/                  # Environment configurations
│   │   ├── grasping_env.yaml
│   │   └── manipulation_env.yaml
│   ├── models/                        # Model configurations
│   │   ├── dqn_config.yaml
│   │   └── policy_gradient_config.yaml
│   └── training/                      # Training configurations
│       ├── training_config.yaml
│       └── hyperparameters.yaml
├── src/                               # Source code
│   ├── __init__.py
│   ├── environments/                  # RL environments
│   │   ├── __init__.py
│   │   ├── base_env.py               # Base environment class
│   │   ├── grasping_env.py           # Grasping environment
│   │   └── utils.py                  # Environment utilities
│   ├── agents/                        # RL agents
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Base agent class
│   │   ├── dqn_agent.py              # DQN implementation
│   │   ├── policy_gradient_agent.py  # Policy gradient methods
│   │   └── replay_buffer.py          # Experience replay
│   ├── models/                        # Neural network models
│   │   ├── __init__.py
│   │   ├── q_network.py              # Q-network architecture
│   │   ├── policy_network.py         # Policy network
│   │   └── value_network.py          # Value function network
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── mujoco_utils.py           # MuJoCo-specific utilities
│   │   ├── visualization.py          # Plotting and visualization
│   │   ├── data_utils.py             # Data processing utilities
│   │   └── metrics.py                # Evaluation metrics
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── evaluator.py              # Model evaluation
│   │   └── checkpoint_manager.py     # Model checkpointing
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── visualize.py                   # Visualization script
│   └── demo.py                        # Demo script
├── models/                            # MuJoCo XML models
│   ├── robot_arm.xml                 # Robot arm with gripper
│   ├── simple_pendulum.xml           # Simple pendulum (for testing)
│   ├── objects/                      # Graspable objects
│   │   ├── cube.xml
│   │   ├── cylinder.xml
│   │   └── sphere.xml
│   └── scenes/                       # Complete scenes
│       ├── grasping_scene.xml
│       └── manipulation_scene.xml
├── examples/                          # Example implementations
│   ├── basic_viewer.py               # Basic MuJoCo viewer
│   ├── simple_controller.py          # Simple control example
│   ├── record_simulation.py          # Simulation recording
│   ├── dqn_example.py
│   └── policy_gradient_example.py
├── notebooks/                         # Jupyter notebooks
├── data/                              # Data storage
│   ├── raw/                          # Raw simulation data
│   ├── processed/                    # Processed training data
│   ├── checkpoints/                  # Model checkpoints
│   └── logs/                         # Training logs
├── results/                           # Results and outputs
│   ├── plots/                        # Generated plots
│   ├── videos/                       # Simulation videos
│   ├── models/                       # Trained models
└── └── reports/                      # Analysis reports
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd coms4733_project
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

### 1. Basic Environment Test
```bash
mjpython examples/basic_viewer.py
```

### 2. Train a DQN Agent
```bash
python src/train.py --config config/training/training_config.yaml
```

### 3. Evaluate a Trained Model
```bash
python src/evaluate.py --checkpoint data/checkpoints/dqn_model.pth
```

### 4. Visualize Training Results
```bash
python src/visualize.py --log_dir data/logs/
```

## Key Components

### Environments
- **GraspingEnvironment**: Main environment for grasping tasks
- **BaseEnvironment**: Abstract base class for all environments
- Configurable object types, reward functions, and observation spaces

### Agents
- **DQNAgent**: Deep Q-Network implementation with experience replay
- **PolicyGradientAgent**: Policy gradient methods (REINFORCE, A2C)
- **ReplayBuffer**: Experience replay buffer for off-policy learning

### Models
- **QNetwork**: Deep Q-Network architecture
- **PolicyNetwork**: Policy network for continuous/discrete actions
- **ValueNetwork**: Value function approximator

### Training
- **Trainer**: Main training loop with logging and checkpointing
- **Evaluator**: Model evaluation and performance metrics
- **CheckpointManager**: Model saving and loading utilities

## Configuration

The project uses YAML configuration files for easy experimentation:

- `config/environments/`: Environment-specific settings
- `config/models/`: Neural network architectures
- `config/training/`: Training hyperparameters

## References

- "Robotic Grasping using Deep Reinforcement Learning" - Reference paper
- MuJoCo Physics Engine Documentation
- Deep Reinforcement Learning Algorithms (DQN, Policy Gradients)