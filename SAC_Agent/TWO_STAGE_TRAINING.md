# Two-Stage Training for SAC Agent

## Overview

This implementation provides a two-stage training approach to speed up the learning process for the Soft Actor-Critic (SAC) agent in the grasping environment. The approach separates pixel-based feature learning from full robot control, allowing faster initial training before fine-tuning on the complete task.

## Why Two-Stage Training?

Training a robot to grasp objects involves:
1. **Learning visual features**: Understanding which pixels correspond to graspable objects
2. **Learning robot control**: Coordinating inverse kinematics, motion planning, and grasping

The two-stage approach:
- **Stage 1 (Fast)**: Focuses on learning visual features without robot motion overhead
- **Stage 2 (Full)**: Fine-tunes on the complete task with robot motion

This separation provides:
- **Faster training**: Stage 1 runs much faster (no IK, no PID loops, no robot motion)
- **Better sample efficiency**: Agent learns visual patterns before dealing with robot dynamics
- **Transfer learning**: Visual features learned in Stage 1 transfer to Stage 2

## Architecture

### Stage 1: Fast Training (`pretain_vision_model.py`)

**Environment**: `FastGraspEnv`
- Inherits from `GraspEnv` but overrides `step()` method
- **No robot motion**: Actions are evaluated instantly without moving the UR5
- **Pixel-based rewards**: Computes reward based on distance from selected pixel to nearest object
- **Contextual bandit**: Each step is independent (done=True after one action)

**Key Features**:
- Same observation space (RGB + depth)
- Same action space (pixel selection + rotation)
- Fast execution: Only renders observation and computes geometric distance
- Reward shaping: `reward = success - shape_scale * distance`

**Training Process**:
1. Agent selects a pixel and rotation
2. Environment computes distance from pixel to nearest object
3. Reward is based on proximity (success if within `success_radius = 0.02m`)
4. No robot motion, no IK, no PID control

### Stage 2: Full Training (`finetune_sac_model.py`)

**Environment**: `Grasper-v0` (full `GraspEnv`)
- Complete robot simulation with UR5 arm
- Full HER (Hindsight Experience Replay) support
- Goal-based rewards
- Robot motion, IK, and PID control

**Training Process**:
1. Loads pretrained actor weights from Stage 1
2. Fine-tunes on full environment with robot motion
3. Uses HER for goal relabeling
4. Learns to coordinate visual understanding with robot control

## Implementation Details

### Mode System

The `SAC_Agent` class supports two modes:

```python
agent = SAC_Agent(
    mode="fast",  # or "full"
    # ... other parameters
)
```

**Fast Mode** (`mode="fast"`):
- `use_her = False` (automatically set)
- Uses `ReplayBuffer` (simple replay, no HER)
- `goal_dim = 0` (no goal conditioning)
- Environment: `FastGraspEnv`

**Full Mode** (`mode="full"`):
- `use_her = True` (automatically set)
- Uses `HERReplayBuffer` (with goal relabeling)
- `goal_dim = 3` (goal conditioning)
- Environment: `Grasper-v0`

### Key Components

#### 1. FastGraspEnv (`gym_grasper/envs/FastGraspEnv.py`)

```python
class FastGraspEnv(GraspEnv):
    def step(self, action):
        # Decode action to pixel coordinates
        # Get depth at that pixel
        # Convert to world coordinates
        # Compute distance to nearest object
        # Return reward based on distance
        # done = True (contextual bandit)
```

**Reward Function**:
- `success = (min_dist < success_radius)`
- `reward = success - shape_scale * min_dist`
- `shape_scale = 5.0` (tunable parameter)

#### 2. ReplayBuffer (`SAC_Agent/ReplayBuffer.py`)

Simple replay buffer for fast mode:
- Stores transitions: `(state, action, next_state, reward, done)`
- No goal information
- No episode tracking
- Standard uniform sampling

#### 3. SAC_Agent Modifications

**Environment Selection**:
```python
if self.mode == "full":
    self.env = gym.make("gym_grasper:Grasper-v0", ...)
else:
    from gym_grasper.envs.FastGraspEnv import FastGraspEnv
    self.env = FastGraspEnv(...)
```

**Memory Setup**:
```python
if self.mode == "full":
    self.memory = HERReplayBuffer(...)
    self.goal_dim = 3
else:
    self.memory = ReplayBuffer(...)
    self.goal_dim = 0
```

**Network Architecture**:
- Networks automatically adapt to `goal_dim`
- Fast mode: `Actor(state)` and `QNetwork(state, action)`
- Full mode: `Actor(state, goal)` and `QNetwork(state, action, goal)`

**Learning**:
- Both modes use the same SAC algorithm
- Fast mode skips goal-related computations
- Networks handle `goal_dim=0` gracefully

## Usage

### Stage 1: Fast Training

```bash
cd SAC_Agent
python train_fast.py
```

**What happens**:
1. Creates `SAC_Agent` with `mode="fast"`
2. Trains on `FastGraspEnv` for 3000 episodes
3. Saves best actor weights to `pretrained_actor_fast.pth`
4. Saves final weights to `pretrained_actor_fast_final.pth`

**Output**:
- `pretrained_actor_fast.pth`: Best model (highest mean reward)
- `pretrained_actor_fast_final.pth`: Final model after all episodes

### Stage 2: Full Training

```bash
cd SAC_Agent
python full_train.py
```

**What happens**:
1. Creates `SAC_Agent` with `mode="full"`
2. Loads `pretrained_actor_fast.pth` (if available)
3. Fine-tunes on full `Grasper-v0` environment
4. Uses HER for goal relabeling
5. Saves checkpoints to `Models/sac_full_best.pt` and `Models/sac_full_final.pt`

**Note**: If `pretrained_actor_fast.pth` is not found, training starts from scratch.

## Configuration

### Stage 1 Parameters (`train_fast.py`)

```python
N_EPISODES = 3000
STEPS_PER_EPISODE = 30
SAVE_WEIGHTS = True
```

**FastGraspEnv Parameters**:
- `success_radius = 0.02` (2 cm): Distance threshold for success
- `shape_scale = 5.0`: Reward shaping coefficient

### Stage 2 Parameters (`full_train.py`)

```python
N_EPISODES = 1000
STEPS_PER_EPISODE = 30
SAVE_WEIGHTS = True
her_ratio = 0.5  # HER relabeling ratio
```

## Training Flow

### Stage 1 Flow

```
Episode Loop:
  Reset environment
  For each step:
    Select action (pixel + rotation)
    Environment computes reward (no robot motion)
    Store transition in ReplayBuffer
    Learn (SAC update)
    If done: break (contextual bandit)
  Save best model if improved
```

### Stage 2 Flow

```
Load pretrained actor weights
Episode Loop:
  Start new episode
  Reset environment
  For each step:
    Select action (pixel + rotation)
    Execute action (robot moves)
    Store transition in episode buffer
    Learn (SAC update with HER)
    If done: break
  Finalize episode (push to HER buffer)
  Save best checkpoint if improved
```

## Benefits

1. **Speed**: Stage 1 runs ~10-100x faster (no robot motion)
2. **Sample Efficiency**: Visual features learned before robot control
3. **Transfer Learning**: Pretrained visual features help Stage 2
4. **Modularity**: Can train Stage 1 independently
5. **Flexibility**: Can skip Stage 1 and train Stage 2 from scratch

## Technical Notes

### Goal Handling

- **Fast Mode**: `goal_dim = 0`, networks ignore goal inputs
- **Full Mode**: `goal_dim = 3`, networks use goal conditioning
- `transform_observation()` returns `None` for goals in fast mode
- Networks handle `None` goals gracefully

### Action Space

Both modes use the same action space:
- `MultiDiscrete([H*W, n_rotations])`
- First component: flattened pixel index
- Second component: rotation index

### Observation Space

Both modes use the same observation structure:
- RGB image: `(H, W, 3)`
- Depth image: `(H, W)`
- Goals: Only in full mode

### Network Compatibility

The pretrained actor from Stage 1 can be loaded into Stage 2 because:
- Both use the same state encoder (ResNet)
- Fast mode actor has `goal_dim=0` but can be extended
- Full mode actor expects `goal_dim=3` but can work with pretrained weights

**Note**: When loading Stage 1 weights into Stage 2, the goal-conditioned layers are randomly initialized, but the visual feature extractor is pretrained.

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `gym_grasper` is in Python path
   ```python
   sys.path.insert(0, str(project_root))
   ```

2. **Missing Pretrained Weights**: Stage 2 will start from scratch if `pretrained_actor_fast.pth` is not found

3. **Goal Dimension Mismatch**: Networks automatically handle `goal_dim=0` vs `goal_dim=3`

4. **Memory Issues**: Reduce `max_possible_samples` or `memory_size` if running out of memory

## Future Improvements

- **Progressive Training**: Gradually increase robot motion complexity
- **Curriculum Learning**: Start with easier objects, progress to harder ones
- **Multi-Task Learning**: Train on multiple object types simultaneously
- **Domain Adaptation**: Fine-tune on different camera angles or lighting

## References

- Soft Actor-Critic (SAC): [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
- Hindsight Experience Replay (HER): [Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)
- Two-Stage Training: Common in computer vision and robotics for efficiency

