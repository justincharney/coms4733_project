# Robotic Grasping with SAC and HER

This project implements a Soft Actor-Critic (SAC) reinforcement learning agent with Hindsight Experience Replay (HER) for robotic grasping tasks using MuJoCo simulation. It also includes a DQN baseline implementation for comparison.

## Overview

The project includes two main approaches:

### SAC Training Pipeline (Two-Stage)
1. **Stage 1: Vision Pretraining** - Fast training on pixel-based rewards (no robot motion)
2. **Stage 2: Fine-tuning** - Full training with robot motion, IK, and HER

### DQN Baseline
- **DQN Training** - Discrete Q-Network baseline with ResNet architecture
- Uses epsilon-greedy exploration and binary cross-entropy loss
- Fast training on pixel-based rewards (no robot motion)

## Installation

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### ⚠️ IMPORTANT: Headless Rendering Setup (Ubuntu/Linux)

**If you're running training on a headless server (no display), you MUST set up Xvfb for MuJoCo rendering to work.**

MuJoCo requires an OpenGL context to render depth images. Without it, all depth values will be zero and training will fail. Use Xvfb (X Virtual Framebuffer) to create a virtual display:

```bash
# Install Xvfb package (Ubuntu/Debian)
sudo apt-get install xvfb

# Run training scripts with xvfb-run
xvfb-run -a python SAC_Agent/pretrain_vision_model.py
xvfb-run -a python SAC_Agent/finetune_sac_model.py
```

**Why this is needed:**
- MuJoCo's renderer requires an OpenGL context
- Headless servers don't have a display by default
- Xvfb creates a virtual X server for rendering
- Without it, depth images will be all zeros and pixel-to-world mapping will fail

**Alternative options (if Xvfb doesn't work):**
```bash
# Option 1: Set DISPLAY if you have X server access
export DISPLAY=:0
python SAC_Agent/finetune_sac_model.py

# Option 2: Use EGL backend (if available)
export MUJOCO_GL=egl
python SAC_Agent/finetune_sac_model.py

# Option 3: Use OSMesa (software rendering)
export MUJOCO_GL=osmesa
python SAC_Agent/finetune_sac_model.py
```

### Project Structure

```
coms4733_project/
├── SAC_Agent/
│   ├── SAC.py                 # SAC agent implementation
│   ├── HER.py                 # Hindsight Experience Replay
│   ├── networks.py            # Actor and Q-network architectures
│   ├── ReplayBuffer.py        # Replay buffer for experience storage
│   ├── training_logger.py     # Training logger with CSV export
│   ├── metrics.py             # Motion quality metrics
│   ├── pretrain_vision_model.py  # Stage 1: Pretraining script
│   ├── finetune_sac_model.py     # Stage 2: Fine-tuning script
│   ├── evaluate_model.py          # Model evaluation script
│   └── plot_training.py           # Plot training curves
├── discrete_dqn/
│   ├── network.py            # DQN ResNet architecture
│   ├── dqn_agent.py          # DQN agent implementation
│   ├── training_logger.py    # Training logger with CSV export
│   ├── eval_utils.py         # Shared evaluation utilities
│   ├── train_dqn.py          # DQN training script
│   ├── evaluate_model.py     # DQN evaluation script
│   └── plot_training.py      # Plot DQN training curves
├── gym_grasper/
│   ├── envs/
│   │   ├── GraspingEnv.py     # Full grasping environment
│   │   └── FastGraspEnv.py    # Fast pretraining environment
│   └── controller/
│       └── MujocoController.py # MuJoCo robot controller
├── Models/                    # Saved model checkpoints
├── logs/                      # Training logs and CSV files
├── evaluation_results/        # Evaluation results
├── demo.py                    # Visual demo of trained model
└── visualize_env.py           # Environment visualization tool
```

## Usage

### 0. Visualize Environment

Before training, you can visualize the grasping environment to understand its structure, workspace boundaries, and camera views.

```bash
# From project root directory

# Basic visualization (display in OpenCV window)
# ⚠️ On headless servers: xvfb-run -a python visualize_env.py
python visualize_env.py

# Save visualization images instead of displaying (no xvfb needed)
python visualize_env.py --mode save
```

**Options:**
- `--mode`: Visualization mode - `"opencv"` to display window (default), `"save"` to save images
- `--steps`: Number of random steps to take (default: 0, just show initial state)
- `--no-workspace`: Disable workspace boundaries visualization
- `--no-reachable`: Disable reachable workspace visualization
- `--single-camera`: Only show top-down camera view (disable multi-camera view)
- `--include-main1`: Include main1 camera (note: often produces blank/blue images)

**What it shows:**
- RGB and depth observations from multiple camera views (top-down, side, etc.)
- Workspace boundaries drawn on images
- Reachable workspace visualization (points where IK succeeds)
- Environment information panel with workspace bounds and goal information

**Output:**
- Interactive OpenCV window (default mode) or saved images (`--mode save`)

### 1. Stage 1: Vision Pretraining

Fast pretraining on pixel-based rewards without robot motion. This stage learns to identify good grasp locations from visual input.

```bash
# From project root directory

# Basic usage (default: 3000 episodes)
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/pretrain_vision_model.py
python SAC_Agent/pretrain_vision_model.py

# Custom configuration
python SAC_Agent/pretrain_vision_model.py \
    --episodes 3000 \
    --save-interval 10 \
    --model-name pretrain_vision
```

**Options:**
- `--episodes`: Number of training episodes (default: 3000)
- `--save-interval`: Save checkpoint every N episodes if improving (default: 10)
- `--model-name`: Name for the model (default: "pretrain_vision")
- `--log-dir`: Directory for log files (default: "SAC_Agent/logs")
- `--models-dir`: Directory for model checkpoints (default: "SAC_Agent/Models")

**Output:**
- Model checkpoints: `SAC_Agent/Models/pretrain_vision_best.pt`, `SAC_Agent/Models/pretrain_vision_final.pt`
- Training logs: `SAC_Agent/logs/pretrain_vision_*.log`
- Training metrics: `SAC_Agent/logs/pretrain_vision_*.csv`

### 2. Stage 2: Fine-tuning

Full training with robot motion, inverse kinematics, and HER. Loads pretrained weights from Stage 1.

```bash
# From project root directory

# With pretrained model from Stage 1
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/finetune_sac_model.py ...
python SAC_Agent/finetune_sac_model.py \
    --pretrained SAC_Agent/Models/pretrain_vision_best.pt

# Custom configuration
python SAC_Agent/finetune_sac_model.py \
    --episodes 1000 \
    --save-interval 10 \
    --model-name sac_finetune \
    --pretrained SAC_Agent/Models/pretrain_vision_best.pt
```

**Options:**
- `--episodes`: Number of training episodes (default: 1000)
- `--save-interval`: Save checkpoint every N episodes if improving (default: 10)
- `--model-name`: Name for the model (default: "sac_finetune")
- `--log-dir`: Directory for log files (default: "SAC_Agent/logs")
- `--models-dir`: Directory for model checkpoints (default: "SAC_Agent/Models")
- `--pretrained`: Path to pretrained model from Stage 1 (optional)

**Output:**
- Model checkpoints: `SAC_Agent/Models/sac_finetune_best.pt`, `SAC_Agent/Models/sac_finetune_final.pt`
- Training logs: `SAC_Agent/logs/sac_finetune_*.log`
- Training metrics: `SAC_Agent/logs/sac_finetune_*.csv`

### 3. Plot Training Curves

Visualize training progress from CSV files generated during training.

```bash
# From project root directory

# Plot SAC training curves (no xvfb needed - just reads CSV files)
python SAC_Agent/plot_training.py SAC_Agent/logs/sac_finetune_20240101_120000.csv

# Save plots to different directory
python SAC_Agent/plot_training.py SAC_Agent/logs/ --output plots/

# Plot DQN training curves (no xvfb needed)
python discrete_dqn/plot_training.py discrete_dqn/logs/dqn_baseline_20240101_120000.csv
```

**Options:**
- `csv_path`: Path to CSV file or directory containing CSV files
- `--output`: Directory to save plots (default: same as CSV file location)
- `--no-show`: Don't display plots interactively, just save them

**Output:**
- Plot files: `SAC_Agent/logs/{model_name}_{timestamp}_training_curves.png`
- Shows: Reward curves, loss curves, success rate, episode length, motion quality metrics

### 4. Evaluate Model

Evaluate a trained model on the grasping environment and compute performance metrics.

```bash
# From project root directory
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/evaluate_model.py ...

# Basic evaluation (100 episodes)
python SAC_Agent/evaluate_model.py --model-path SAC_Agent/Models/sac_finetune_best.pt

# With rendering (visualize evaluation)
python SAC_Agent/evaluate_model.py \
    --model-path SAC_Agent/Models/sac_finetune_best.pt \
    --episodes 100 \
    --render
```

**Options:**
- `--model-path`: Path to saved model checkpoint (.pt file) **[required]**
- `--episodes`: Number of evaluation episodes (default: 100)
- `--render`: Render the environment during evaluation
- `--no-save`: Don't save results to CSV
- `--results-dir`: Directory to save evaluation results (default: "SAC_Agent/evaluation_results")
- `--max-episode-steps`: Maximum steps per episode (default: 100)

**Output:**
- Console: Summary statistics (success rate, rewards, episode lengths, motion quality)
- CSV file: `SAC_Agent/evaluation_results/evaluation_{model_name}_{timestamp}.csv` (if saved)

### 5. Visual Demo

Run a visual demonstration of the trained model performing grasping tasks.

```bash
# From project root directory

# Basic demo (5 episodes with rendering)
# ⚠️ On headless servers: xvfb-run -a python demo.py ...
python demo.py --model-path SAC_Agent/Models/sac_finetune_best.pt

# Without rendering (faster, no xvfb needed)
python demo.py \
    --model-path SAC_Agent/Models/sac_finetune_best.pt \
    --episodes 5 \
    --no-render
```

**Options:**
- `--model-path`: Path to saved model checkpoint (.pt file) **[required]**
- `--episodes`: Number of demo episodes to run (default: 5)
- `--no-render`: Disable rendering (faster but no visualization)
- `--max-episode-steps`: Maximum steps per episode (default: 100)
- `--delay`: Delay in seconds between episodes (default: 2.0)

**Output:**
- Visual demonstration of the robot performing grasping tasks
- Console: Episode-by-episode results and summary statistics

### 6. DQN Baseline Training

Train the DQN baseline network on FastGraspEnv using epsilon-greedy exploration.

```bash
# From project root directory

# Basic usage (default: 1480 episodes as per milestone)
# ⚠️ On headless servers: xvfb-run -a python discrete_dqn/train_dqn.py
python discrete_dqn/train_dqn.py

# Custom configuration
python discrete_dqn/train_dqn.py \
    --episodes 2000 \
    --save-interval 10 \
    --model-name dqn_baseline \
    --batch-size 32
```

**Options:**
- `--episodes`: Number of training episodes (default: 1480)
- `--save-interval`: Save checkpoint every N episodes if improving (default: 10)
- `--model-name`: Name for the model (default: "dqn_baseline")
- `--log-dir`: Directory for log files (default: "discrete_dqn/logs")
- `--models-dir`: Directory for model checkpoints (default: "discrete_dqn/Models")
- `--batch-size`: Batch size for training (default: 32)

**Output:**
- Model checkpoints: `discrete_dqn/Models/dqn_baseline_best.pt`, `discrete_dqn/Models/dqn_baseline_final.pt`
- Training logs: `discrete_dqn/logs/dqn_baseline_*.log`
- Training metrics: `discrete_dqn/logs/dqn_baseline_*.csv`

### 7. DQN Evaluation

Evaluate a trained DQN model on the grasping environment.

```bash
# From project root directory
# ⚠️ On headless servers: xvfb-run -a python discrete_dqn/evaluate_model.py ...

# Basic evaluation (100 episodes)
python discrete_dqn/evaluate_model.py --model-path discrete_dqn/Models/dqn_baseline_best.pt

# With rendering (visualize evaluation)
python discrete_dqn/evaluate_model.py \
    --model-path discrete_dqn/Models/dqn_baseline_best.pt \
    --episodes 100 \
    --render

# With full environment for motion quality metrics (jerk, acceleration)
# Note: This uses GraspingEnv which actually moves the robot (slower but provides motion metrics)
python discrete_dqn/evaluate_model.py \
    --model-path discrete_dqn/Models/dqn_baseline_best.pt \
    --episodes 100 \
    --use-full-env
```

**Options:**
- `--model-path`: Path to saved model checkpoint (.pt file) **[required]**
- `--episodes`: Number of evaluation episodes (default: 100)
- `--render`: Render the environment during evaluation
- `--no-save`: Don't save results to CSV
- `--results-dir`: Directory to save evaluation results (default: "discrete_dqn/evaluation_results")
- `--use-full-env`: Use `GraspingEnv` instead of `FastGraspEnv` for evaluation. **Required for meaningful motion quality metrics** (jerk, acceleration). Note: This will be slower as it actually moves the robot. Without this flag, motion metrics will be zero because `FastGraspEnv` doesn't move the robot.
- `--max-episode-steps`: Maximum steps per episode (default: 100). Note: `FastGraspEnv` always terminates after 1 step, so this mainly applies when using `--use-full-env`.

**Output:**
- Console: Summary statistics (success rate, rewards, episode lengths, motion quality metrics)
- CSV file: `discrete_dqn/evaluation_results/evaluation_{model_name}_{timestamp}.csv` (if saved)

**Note on Motion Quality Metrics:**
- By default, DQN evaluation uses `FastGraspEnv` which doesn't move the robot (fast pixel-based rewards only)
- Motion quality metrics (jerk, acceleration) require actual robot motion, so they will be zero with `FastGraspEnv`
- Use `--use-full-env` flag to evaluate with `GraspingEnv` which moves the robot and provides meaningful motion metrics
- Collision detection works with both environments since it doesn't require motion

### 8. DQN Plot Training Curves

Visualize DQN training progress from CSV files.

```bash
# From project root directory

# Plot a specific CSV file (no xvfb needed)
python discrete_dqn/plot_training.py discrete_dqn/logs/dqn_baseline_20240101_120000.csv

# Plot all CSV files in directory
python discrete_dqn/plot_training.py discrete_dqn/logs/

# Save plots to different directory
python discrete_dqn/plot_training.py discrete_dqn/logs/ --output plots/
```

**Options:**
- `csv_path`: Path to CSV file or directory containing CSV files
- `--output`: Directory to save plots (default: same as CSV file location)
- `--no-show`: Don't display plots interactively, just save them

**Output:**
- Plot files: `discrete_dqn/logs/{model_name}_{timestamp}_training_curves.png`
- Shows: Reward curves, loss curves, success rate, epsilon decay, Q-value statistics

## Complete Training Pipeline Examples

### SAC Training Pipeline

```bash
# From project root directory

# Step 0: (Optional) Visualize environment to understand structure
# ⚠️ On headless servers: xvfb-run -a python visualize_env.py
python visualize_env.py

# Step 1: Pretrain vision model
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/pretrain_vision_model.py
python SAC_Agent/pretrain_vision_model.py --episodes 3000

# Step 2: Fine-tune with full robot motion
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/finetune_sac_model.py
python SAC_Agent/finetune_sac_model.py \
    --episodes 1000 \
    --pretrained SAC_Agent/Models/pretrain_vision_best.pt

# Step 3: Plot training curves (no xvfb needed)
python SAC_Agent/plot_training.py SAC_Agent/logs/

# Step 4: Evaluate the model
# ⚠️ On headless servers: xvfb-run -a python SAC_Agent/evaluate_model.py ...
python SAC_Agent/evaluate_model.py \
    --model-path SAC_Agent/Models/sac_finetune_best.pt \
    --episodes 100

# Step 5: Run visual demo
# ⚠️ On headless servers: xvfb-run -a python demo.py ...
python demo.py \
    --model-path SAC_Agent/Models/sac_finetune_best.pt \
    --episodes 10
```

### DQN Baseline Training Pipeline

```bash
# From project root directory

# Step 1: Train DQN baseline
# ⚠️ On headless servers: xvfb-run -a python discrete_dqn/train_dqn.py
python discrete_dqn/train_dqn.py --episodes 1480

# Step 2: Plot training curves (no xvfb needed)
python discrete_dqn/plot_training.py discrete_dqn/logs/

# Step 3: Evaluate the model
# ⚠️ On headless servers: xvfb-run -a python discrete_dqn/evaluate_model.py ...
python discrete_dqn/evaluate_model.py \
    --model-path discrete_dqn/Models/dqn_baseline_best.pt \
    --episodes 100

# Step 3b: (Optional) Evaluate with full environment for motion quality metrics
# ⚠️ On headless servers: xvfb-run -a python discrete_dqn/evaluate_model.py ...
python discrete_dqn/evaluate_model.py \
    --model-path discrete_dqn/Models/dqn_baseline_best.pt \
    --episodes 100 \
    --use-full-env
```

## Key Features

### Reward System
- **Grasp Success**: +5.0 reward when successfully grasping an object
- **Reaching**: Distance-based shaping reward to guide the agent
- **HER Integration**: Rewards only given when position is close AND object is grasped

### Hindsight Experience Replay (HER)
- **Strategy**: 'future' (default), 'final', or 'episode'
- **Grasp Verification**: HER only rewards transitions where objects were actually grasped
- **Goal Relabeling**: Failed episodes are relabeled with achieved goals

### Motion Quality Metrics
- Mean jerk (smoothness)
- RMS acceleration
- Collision detection
- Vertical overshoot tracking

**Note:** Motion quality metrics (jerk, acceleration) require actual robot motion. They are only meaningful when:
- Evaluating SAC models (which use `GraspingEnv` with robot motion)
- Evaluating DQN models with `--use-full-env` flag (uses `GraspingEnv` instead of `FastGraspEnv`)

When using `FastGraspEnv` (default for DQN evaluation), motion metrics will be zero because the robot doesn't move. Collision detection works with both environments.

## File Outputs

### SAC Training
- **Models**: `SAC_Agent/Models/{model_name}_best.pt`, `SAC_Agent/Models/{model_name}_final.pt`
- **Logs**: `SAC_Agent/logs/{model_name}_{timestamp}.log`
- **Metrics CSV**: `SAC_Agent/logs/{model_name}_{timestamp}.csv`

### SAC Evaluation
- **Results CSV**: `SAC_Agent/evaluation_results/evaluation_{model_name}_{timestamp}.csv`

### SAC Visualization
- **Training Curves**: `SAC_Agent/logs/{model_name}_{timestamp}_training_curves.png`

### DQN Training
- **Models**: `discrete_dqn/Models/{model_name}_best.pt`, `discrete_dqn/Models/{model_name}_final.pt`
- **Logs**: `discrete_dqn/logs/{model_name}_{timestamp}.log`
- **Metrics CSV**: `discrete_dqn/logs/{model_name}_{timestamp}.csv`

### DQN Evaluation
- **Results CSV**: `discrete_dqn/evaluation_results/evaluation_{model_name}_{timestamp}.csv`

### DQN Visualization
- **Training Curves**: `discrete_dqn/logs/{model_name}_{timestamp}_training_curves.png`

## Troubleshooting

### Model Loading Errors
- Ensure model path is correct
- Check that model was trained with same configuration (goal_dim, action dimensions)

### Environment Errors
- Verify MuJoCo XML files are in correct location
- Check that workspace bounds are valid
- Ensure camera setup is correct

### Rendering Issues

**⚠️ CRITICAL: Headless Server Setup**

If you see errors like:
```
GLFWError: X11: The DISPLAY environment variable is missing
Renderer unavailable (falling back to blank frames)
[WARNING] Depth image is all zeros!
```

**You MUST install and use Xvfb:**

```bash
# Install Xvfb
sudo apt-get install xvfb

# Run training with xvfb-run
xvfb-run -a python SAC_Agent/finetune_sac_model.py
```

**Why this happens:**
- MuJoCo requires OpenGL/display access to render depth images
- Headless servers don't have a display by default
- Without proper rendering, depth values are all zeros
- This causes pixel-to-world mapping to fail and all actions are rejected

**Other rendering options:**
- For servers with X server: `export DISPLAY=:0`
- For EGL support: `export MUJOCO_GL=egl`
- For software rendering: `export MUJOCO_GL=osmesa`
- Note: `--no-render` flag only disables visualization, not depth rendering (which is required for training)

## Command Reference: When to Use xvfb-run

**Commands that NEED `xvfb-run -a` (on headless servers):**
- `python SAC_Agent/pretrain_vision_model.py` - Uses FastGraspEnv (needs rendering for depth)
- `python SAC_Agent/finetune_sac_model.py` - Uses GraspEnv (needs rendering for depth)
- `python SAC_Agent/evaluate_model.py` - Uses GraspEnv (needs rendering for depth)
- `python discrete_dqn/train_dqn.py` - Uses FastGraspEnv (needs rendering for depth)
- `python discrete_dqn/evaluate_model.py` - Uses FastGraspEnv (needs rendering for depth)
- `python visualize_env.py` - Uses OpenCV window (needs display)
- `python demo.py` - Uses rendering (needs display)

**Commands that DON'T need `xvfb-run` (no rendering required):**
- `python SAC_Agent/plot_training.py` - Just reads CSV files and plots
- `python discrete_dqn/plot_training.py` - Just reads CSV files and plots
- `python visualize_env.py --mode save` - Saves images without display
- `python demo.py --no-render` - Disables rendering

**Note:** Even with `--no-render` flags, MuJoCo still needs OpenGL context for depth rendering. The `--no-render` flag only disables visualization windows, not depth image computation.

## Notes

- **Deterministic actions during evaluation**: When evaluating a trained model (using `evaluate_model.py` or `demo.py`), the agent uses deterministic actions (always picks the most likely action) rather than sampling from the action distribution. This provides consistent, reproducible results and represents the agent's best performance without exploration noise.
- Progress bars (tqdm) show real-time metrics during training
- All metrics are automatically logged to CSV for analysis
- HER is automatically disabled for fast pretraining mode (goal_dim=0)
- DQN baseline uses epsilon-greedy exploration and binary cross-entropy loss (gamma=0, no target network)
