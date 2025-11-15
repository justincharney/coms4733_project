import os

os.environ["MUJOCO_GL"] = "osmesa"

import gym
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Modules import ReplayBuffer
import numpy as np
import pickle
import random
import copy
from collections import deque, defaultdict, namedtuple
import time
from .networks import Actor, QNetwork
from pathlib import Path

# Transition with done flag for SAC
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Find project root (parent of SAC_Agent directory)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_OUTPUT_FOLDER = PROJECT_ROOT / "Models"
MODEL_OUTPUT_FOLDER.mkdir(exist_ok=True)


class SAC_Agent:
    """
    Soft Actor-Critic (SAC) agent for discrete action spaces.
    Compatible with the existing grasping environment.
    """

    def __init__(
        self,
        seed,
        n_episodes=1,
        steps_per_episode=1,
        number_accumulations_before_update=1,
        max_possible_samples=1,
        depth_only=False,
        load_path=None,
        train=True,
        auto_alpha=True,
    ):
        """
        Args:
            height: Observation height (in pixels).
            width: Observation width (in pixels).
            learning_rate: Learning rate for optimizers.
            mem_size: Number of transitions to be stored in the replay buffer.
            depth_only: If True, use only depth channel.
            load_path: Path to load pretrained weights.
            train: If True, initialize for training (replay buffer, optimizers).
            seed: Random seed.
            optimizer: Optimizer type ("ADAM" or "SGD").
            alpha: Temperature parameter for entropy regularization.
            gamma: Discount factor.
            tau: Soft update coefficient for target networks.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.depth_only = depth_only
        self.auto_alpha = auto_alpha
        self.height = 200
        self.width = 200
        self.memory_size = 2000
        self.gamma = 0.99  # SAC typically uses non-zero gamma
        self.learning_rate = 0.0003  # SAC default learning rate
        self.tau = 0.005  # Soft update coefficient for target networks
        self.alpha = 0.2  # Initial temperature parameter (entropy regularization)
        self.alpha_lr = 0.0003  # Learning rate for alpha
        self.model = "SAC_RESNET"
        self.algorithm = "SAC"
        self.optimizer = "ADAM"
        self.n_episodes = n_episodes
        self.steps_per_episode = steps_per_episode
        self.max_possible_samples = max_possible_samples
        self.number_accumulations_before_update = number_accumulations_before_update
        self.batch_size = self.max_possible_samples * self.number_accumulations_before_update

        if train:
            self.env = gym.make(
                "gym_grasper:Grasper-v0",
                image_height=self.height,
                image_width=self.width,
                show_obs=False,
                render=False,
            )
        else:
            self.env = gym.make(
                "gym_grasper:Grasper-v0",
                image_height=self.height,
                image_width=self.width,
                show_obs=False,
                demo=True,
                render=True,
            )

        self.n_actions_1, self.n_actions_2 = (
            self.env.action_space.nvec[0],
            self.env.action_space.nvec[1],
        )
        self.output = self.n_actions_1 * self.n_actions_2

        # Automatic alpha tuning (after n_actions is defined)
        if auto_alpha:
            # Learnable log_alpha parameter
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(np.log(self.alpha), device=device), requires_grad=True
            )
            # Target entropy: -log(1/num_actions) for uniform policy baseline
            # For discrete uniform: H = -sum(1/A * log(1/A)) = log(A)
            # We'll use a fraction of this as target (e.g., 0.1 * log(A))
            self.target_entropy = -0.1 * np.log(self.output)
        else:
            # Fixed alpha
            self.log_alpha = None
            self.target_entropy = None

        # Initialize networks
        self.actor = Actor(self.output, depth_only=self.depth_only).to(device)
        self.critic1 = QNetwork(self.output, depth_only=self.depth_only).to(device)
        self.critic2 = QNetwork(self.output, depth_only=self.depth_only).to(device)

        # Target networks
        self.critic1_target = QNetwork(self.output, depth_only=self.depth_only).to(device)
        self.critic2_target = QNetwork(self.output, depth_only=self.depth_only).to(device)

        # Initialize target networks with same weights as main networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Load weights if provided
        if load_path is not None:
            checkpoint = torch.load(load_path)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
            self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
            self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
            self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
            print("Successfully loaded weights from {}.".format(load_path))

        # Set up transforms
        self.normal_rgb = T.Compose(
            [
                T.ToPILImage(),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                T.ToTensor(),
            ]
        )
        self.normal_rgb_no_jitter_no_noise = T.Compose([T.ToTensor()])
        self.normal_depth = T.Compose(
            [T.Lambda(lambda x: x + 0.01 * torch.randn_like(x))]
        )
        self.depth_threshold = np.round(
            self.env.model.cam_pos0[self.env.model.camera_name2id("top_down")][2] - self.env.TABLE_HEIGHT + 0.01,
            decimals=3,
        )
        self.last_action = None

        if train:
            # Set up replay buffer
            self.memory = ReplayBuffer(self.memory_size, simple=False)

            # Override push to handle done flag
            def push_with_done(memory_self, state, action, next_state, reward, done):
                # Store transition with done flag
                if len(memory_self.memory) < memory_self.size:
                    memory_self.memory.append(None)
                memory_self.memory[memory_self.position] = Transition(
                    state, action, next_state, reward, done
                )
                memory_self.position = (memory_self.position + 1) % memory_self.size

            # Bind the method to the memory object
            import types
            self.memory.push = types.MethodType(push_with_done, self.memory)

            # Optimizers
            if self.optimizer == "SGD":
                self.actor_optimizer = optim.SGD(
                    self.actor.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=0.00002,
                )
                self.critic1_optimizer = optim.SGD(
                    self.critic1.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=0.00002,
                )
                self.critic2_optimizer = optim.SGD(
                    self.critic2.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=0.00002,
                )
            elif self.optimizer == "ADAM":
                self.actor_optimizer = optim.Adam(
                    self.actor.parameters(), lr=self.learning_rate, weight_decay=0.00002
                )
                self.critic1_optimizer = optim.Adam(
                    self.critic1.parameters(), lr=self.learning_rate, weight_decay=0.00002
                )
                self.critic2_optimizer = optim.Adam(
                    self.critic2.parameters(), lr=self.learning_rate, weight_decay=0.00002
                )

            # Alpha optimizer (only if auto_alpha is enabled)
            if auto_alpha:
                self.alpha_optimizer = optim.Adam(
                    [self.log_alpha], lr=self.alpha_lr, weight_decay=0.0
                )

            if load_path is not None:
                self.actor_optimizer.load_state_dict(
                    checkpoint["actor_optimizer_state_dict"]
                )
                self.critic1_optimizer.load_state_dict(
                    checkpoint["critic1_optimizer_state_dict"]
                )
                self.critic2_optimizer.load_state_dict(
                    checkpoint["critic2_optimizer_state_dict"]
                )
                if auto_alpha and "alpha_optimizer_state_dict" in checkpoint:
                    self.alpha_optimizer.load_state_dict(
                        checkpoint["alpha_optimizer_state_dict"]
                    )
                    if "log_alpha" in checkpoint:
                        self.log_alpha.data = checkpoint["log_alpha"]
                self.steps_done = (
                    checkpoint["step"] if "step" in checkpoint.keys() else 0
                )
            else:
                self.steps_done = 0

            # Create description and weight path
            if load_path is None:
                date = "_".join(
                    [
                        str(time.localtime()[1]),
                        str(time.localtime()[2]),
                        str(time.localtime()[0]),
                        str(time.localtime()[3]),
                        str(time.localtime()[4]),
                    ]
                )
                self.DESCRIPTION = "_".join(
                    [
                        self.algorithm,
                        self.model,
                        "LR",
                        str(self.learning_rate),
                        "OPTIM",
                        self.optimizer,
                        "H",
                        str(self.height),
                        "W",
                        str(self.width),
                        "STEPS",
                        str(self.n_episodes * self.steps_per_episode),
                        "BUFFER_SIZE",
                        str(self.memory_size),
                        "BATCH_SIZE",
                        str(self.batch_size),
                        "SEED",
                        str(seed),
                    ]
                )
                # Create path (absolute for saving, but we'll display relative)
                weight_filename = self.DESCRIPTION + "_" + date + "_weights.pt"
                self.WEIGHT_PATH = MODEL_OUTPUT_FOLDER / weight_filename
            else:
                self.DESCRIPTION = (
                    "_continue_" + load_path[:-11] + "_at_" + str(self.steps_done)
                )
                self.WEIGHT_PATH = load_path

            # Tensorboard setup
            self.writer = SummaryWriter(comment=self.DESCRIPTION)
            if not self.depth_only:
                self.writer.add_graph(
                    self.actor,
                    torch.zeros(1, 4, self.width, self.height).to(device),
                )
            else:
                self.writer.add_graph(
                    self.actor,
                    torch.zeros(1, 1, self.width, self.height).to(device),
                )

            self.last_1000_rewards = deque(maxlen=1000)
            self.last_100_loss = deque(maxlen=100)
            self.last_1000_actions = deque(maxlen=1000)
            self.greedy_rotations = defaultdict(int)
            self.greedy_rotations_successes = defaultdict(int)
            self.random_rotations_successes = defaultdict(int)

    def select_action(self, state, deterministic=False):
        """
        Select an action using the actor policy.

        Args:
            state: Current state tensor
            deterministic: If True, select greedy action. If False, sample from policy.

        Returns:
            action: Action index
            log_prob: Log probability of the action (for training)
        """
        with torch.no_grad():
            logits = self.actor(state.to(device))
            if deterministic:
                # Greedy action
                action = logits.argmax(dim=1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                # Sample from policy
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def greedy(self, state, return_q_value=True):
        """
        Always returns the greedy action. For demonstrating learned behaviour.

        Args:
            state: Input state tensor
            return_q_value: If True, also compute and return Q-value (slower).
                           If False, uses max logit as approximate Q-value (faster).
        """
        self.last_action = "greedy"
        with torch.no_grad():
            logits = self.actor(state.to(device))
            action = logits.argmax(dim=1)
            if return_q_value:
                # Get Q-value from critic for logging (slower - runs Perception_Module twice)
                q_value = self.critic1(state.to(device), action.to(device))
                return action.cpu(), q_value.item()
            else:
                # Skip critic computation for faster inference
                # Use max logit as approximate Q-value (not true Q-value, but faster)
                q_value = logits.max(dim=1)[0]
                return action.cpu(), q_value.item()

    def transform_observation(self, observation, normalize=True, jitter_and_noise=True):
        """
        Takes an observation dictionary, transforms it into a normalized tensor.
        Same as DQN agent.
        """
        depth = copy.deepcopy(observation["depth"])
        depth[depth > self.depth_threshold] = self.depth_threshold

        if normalize:
            if not self.depth_only:
                rgb = copy.deepcopy(observation["rgb"])

            rng = np.random.default_rng(seed=None)
            depth += rng.normal(loc=0, scale=0.001, size=depth.shape)
            depth *= -1
            depth_min = np.min(depth)
            depth_max = np.max(depth)
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            rgb = observation["rgb"].astype(np.float32)

        depth = np.expand_dims(depth, 0)
        if not self.depth_only:
            if normalize and jitter_and_noise:
                rgb_tensor = self.normal_rgb(rgb).float()
            if normalize and not jitter_and_noise:
                rgb_tensor = self.normal_rgb_no_jitter_no_noise(rgb).float()
            if not normalize:
                self.means, self.stds = self.get_mean_std()
                self.standardize_rgb = T.Compose(
                    [T.ToTensor(), T.Normalize(self.means[0:3], self.stds[0:3])]
                )
                rgb_tensor = self.standardize_rgb(rgb).float()

        depth_tensor = torch.tensor(depth).float()
        if not normalize:
            self.standardize_depth = T.Compose(
                [
                    T.Normalize(self.means[3], self.stds[3]),
                    T.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                ]
            )
            depth_tensor = self.standardize_depth(depth_tensor)

        if not self.depth_only:
            obs_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)
        else:
            obs_tensor = depth_tensor.detach().clone()

        obs_tensor.unsqueeze_(0)
        if not self.depth_only:
            del rgb, depth, rgb_tensor, depth_tensor
        else:
            del depth, depth_tensor

        return obs_tensor

    def get_mean_std(self):
        """Reads and returns the mean and standard deviation values."""
        with open("mean_and_std", "rb") as file:
            raw = file.read()
            values = pickle.loads(raw)
        return values[0:4], values[4:8]

    def transform_action(self, action):
        """Convert action index to multi-discrete action format."""
        action_value = action.item()
        action_1 = action_value % self.n_actions_1
        action_2 = action_value // self.n_actions_1
        return np.array([action_1, action_2])

    def soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def learn(self):
        """
        SAC learning step.
        Updates actor and both critics.
        """
        if len(self.memory) < 2 * self.batch_size:
            print("Filling the replay buffer ...")
            return

        # Sample batch from replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Gradient accumulation
        for i in range(self.number_accumulations_before_update):
            start_idx = i * self.max_possible_samples
            end_idx = (i + 1) * self.max_possible_samples

            state_batch = torch.cat(batch.state[start_idx:end_idx]).to(device)
            action_batch = torch.cat(batch.action[start_idx:end_idx]).to(device)
            next_state_batch = torch.cat(
                batch.next_state[start_idx:end_idx]
            ).to(device)
            reward_batch = torch.cat(batch.reward[start_idx:end_idx]).to(device)
            done_batch = torch.cat(batch.done[start_idx:end_idx]).to(device)

            # FIX: ensure reward and done are shaped (B, 1) to match Q(s,a)
            reward_batch = reward_batch.view(-1, 1)
            done_batch = done_batch.view(-1, 1)

            # Convert action from tensor to indices
            action_indices = action_batch.squeeze().long()

            # Get current alpha (either fixed or from log_alpha)
            if self.auto_alpha:
                current_alpha = self.log_alpha.exp()
            else:
                current_alpha = self.alpha

            # --------- Critic update ---------
            with torch.no_grad():
                # Get next action and log prob from current policy
                next_logits = self.actor(next_state_batch)
                next_dist = torch.distributions.Categorical(logits=next_logits)
                next_action = next_dist.sample()
                next_log_prob = next_dist.log_prob(next_action)

                # Compute target Q-values using target networks
                target_q1 = self.critic1_target(next_state_batch, next_action)
                target_q2 = self.critic2_target(next_state_batch, next_action)
                target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_prob.unsqueeze(
                    1
                )
                # Mask target Q for terminal states
                target_q = reward_batch + self.gamma * (1 - done_batch) * target_q

            # Current Q-values
            current_q1 = self.critic1(state_batch, action_indices)
            current_q2 = self.critic2(state_batch, action_indices)

            # Critic losses
            critic1_loss = (
                F.mse_loss(current_q1, target_q)
                / self.number_accumulations_before_update
            )
            critic2_loss = (
                F.mse_loss(current_q2, target_q)
                / self.number_accumulations_before_update
            )

            # Update critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # --------- Actor update ---------
            logits = self.actor(state_batch)
            dist = torch.distributions.Categorical(logits=logits)
            new_action = dist.sample()
            new_log_prob = dist.log_prob(new_action)

            # Compute Q-values for new actions
            q1_new = self.critic1(state_batch, new_action)
            q2_new = self.critic2(state_batch, new_action)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss: maximize Q - alpha * log_prob (entropy regularization)
            actor_loss = (
                (current_alpha * new_log_prob.unsqueeze(1) - q_new).mean()
                / self.number_accumulations_before_update
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.auto_alpha:
                alpha_loss = (
                    -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
                    / self.number_accumulations_before_update
                )

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Clamp log_alpha to prevent it from becoming too extreme
                with torch.no_grad():
                    self.log_alpha.clamp_(-10, 10)

                # Log alpha value to tensorboard
                if self.steps_done % 10 == 0:
                    self.writer.add_scalar(
                        "Alpha/Value",
                        current_alpha.item(),
                        global_step=self.steps_done,
                    )

            # Soft update target networks
            self.soft_update(self.critic1_target, self.critic1, self.tau)
            self.soft_update(self.critic2_target, self.critic2, self.tau)

            self.last_100_loss.append((critic1_loss.item() + critic2_loss.item()) / 2)

        self.steps_done += 1

    def update_tensorboard(self, reward, action):
        """Update tensorboard metrics."""
        rotation_action = action[1]
        self.last_1000_actions.append(rotation_action)
        if self.last_action == "greedy":
            self.greedy_rotations[str(rotation_action)] += 1
            if reward == 1:
                self.greedy_rotations_successes[str(rotation_action)] += 1
        else:
            if reward == 1:
                self.random_rotations_successes[str(rotation_action)] += 1

        if self.steps_done % 1000 == 0:
            self.writer.add_histogram(
                "Rotation action distribution/Last1000",
                np.array(self.last_1000_actions),
                global_step=self.steps_done,
                bins=[i for i in range(self.n_actions_2)],
            )

        if self.steps_done % 10 == 0:
            self.writer.add_scalars(
                "Total number of rotation actions/Greedy",
                self.greedy_rotations,
                self.steps_done,
            )
            self.writer.add_scalars(
                "Total number of successful rotation actions/Greedy",
                self.greedy_rotations_successes,
                self.steps_done,
            )
            self.writer.add_scalars(
                "Total number of successful rotation actions/Random",
                self.random_rotations_successes,
                self.steps_done,
            )

        self.last_1000_rewards.append(reward)

        if len(self.last_1000_rewards) > 99 and self.steps_done % 10 == 0:
            last_100 = list(self.last_1000_rewards)[-100:]
            mean_reward_100 = np.mean(last_100)
            self.writer.add_scalar(
                "Mean reward/Last100", mean_reward_100, global_step=self.steps_done
            )

        if len(self.last_1000_rewards) > 999 and self.steps_done % 10 == 0:
            mean_reward_1000 = np.mean(self.last_1000_rewards)
            self.writer.add_scalar(
                "Mean reward/Last1000",
                mean_reward_1000,
                global_step=self.steps_done,
            )

        if len(self.last_100_loss) > 99 and self.steps_done % 10 == 0:
            self.writer.add_scalar(
                "Mean loss/Last100",
                np.mean(self.last_100_loss),
                global_step=self.steps_done,
            )
