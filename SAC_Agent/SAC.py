import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from .networks import QNetwork, ActorNetwork
from .ReplayBuffer import ReplayBuffer
from .HER import HindsightExperienceReplay


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm with Hindsight Experience Replay (HER).
    """

    def __init__(
        self,
        rgb_shape=(200, 200, 3),
        depth_shape=(200, 200),
        goal_dim=3,
        pixel_action_dim=40000,
        rotation_action_dim=6,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        buffer_size=100000,
        her_strategy='future',
        her_k=4,
    ):
        """
        Initialize SAC agent.

        Args:
            rgb_shape: Shape of RGB image
            depth_shape: Shape of depth image
            goal_dim: Dimension of goal
            pixel_action_dim: Number of pixel positions
            rotation_action_dim: Number of rotation options
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter (entropy coefficient)
            auto_alpha: Whether to automatically tune alpha
            buffer_size: Size of replay buffer
            her_strategy: HER strategy ('future', 'final', 'episode')
            her_k: Number of HER goals per transition
            device: Device to run on
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha

        # Initialize replay buffer and HER
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.her = HindsightExperienceReplay(self.replay_buffer, strategy=her_strategy, k=her_k)

        # Initialize networks
        # Twin Q-networks
        self.q_network_1 = QNetwork(
            rgb_shape, depth_shape, goal_dim,
            pixel_action_dim, rotation_action_dim
        ).to(self.device)
        self.q_network_2 = QNetwork(
            rgb_shape, depth_shape, goal_dim,
            pixel_action_dim, rotation_action_dim
        ).to(self.device)

        # Target Q-networks
        self.target_q_network_1 = copy.deepcopy(self.q_network_1)
        self.target_q_network_2 = copy.deepcopy(self.q_network_2)

        # Actor network
        self.actor_network = ActorNetwork(
            rgb_shape, depth_shape, goal_dim,
            pixel_action_dim, rotation_action_dim
        ).to(self.device)
        self.goal_dim = goal_dim  # Store for later use

        # Optimizers
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr, weight_decay=0.0)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr, weight_decay=0.0)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr, weight_decay=0.0)

        # Alpha (temperature) tuning
        if auto_alpha:
            # Target entropy for joint action (pixel, rotation)
            # log_probs in actor network is sum of pixel_log_prob + rotation_log_prob
            self.target_entropy = np.log(pixel_action_dim * rotation_action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr, weight_decay=0.0)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        # Episode buffer for HER
        self.episode_buffer = []

    def select_action(self, observation, deterministic=False):
        """
        Select an action given an observation.

        Args:
            observation: Dict with 'rgb', 'depth', 'desired_goal', 'achieved_goal', 'action_mask'
            deterministic: If True, select deterministic action

        Returns:
            Action tuple (pixel_position, rotation)
        """
        # Convert to tensors and validate
        rgb_np = np.array(observation['rgb'])
        depth_np = np.array(observation['depth'])

        # Replace NaN/Inf with valid values
        if np.isnan(rgb_np).any() or np.isinf(rgb_np).any():
            rgb_np = np.nan_to_num(rgb_np, nan=0.0, posinf=255.0, neginf=0.0)
        if np.isnan(depth_np).any() or np.isinf(depth_np).any():
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=10.0, neginf=0.0)

        rgb = torch.FloatTensor(rgb_np).to(self.device)
        depth = torch.FloatTensor(depth_np).to(self.device)

        # Handle goal_dim=0 for fast training mode
        if hasattr(self.actor_network, 'goal_dim') and self.actor_network.goal_dim == 0:
            # Use dummy goals when goal_dim=0
            desired_goal = torch.zeros(3, device=self.device)
            achieved_goal = torch.zeros(3, device=self.device)
        else:
            desired_goal = torch.FloatTensor(observation['desired_goal']).to(self.device)
            achieved_goal = torch.FloatTensor(observation['achieved_goal']).to(self.device)

        # Get action mask if available
        action_mask = observation.get('action_mask', None)

        action = self.actor_network.get_action(
            rgb, depth, desired_goal, achieved_goal, deterministic, action_mask
        )

        return action

    def store_transition(self, state, action, reward, next_state, done, info):
        """
        Store a transition in the episode buffer.
        Will be added to replay buffer when episode ends.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            info: Additional info (for HER)
        """
        self.episode_buffer.append((state, action, reward, next_state, done, info))

    def end_episode(self):
        """
        End episode and store transitions with HER.
        Skips HER if goal_dim=0 (fast training mode).
        """
        if len(self.episode_buffer) > 0:
            # Skip HER for fast training mode (goal_dim=0)
            if self.goal_dim > 0:
                self.her.store_episode(self.episode_buffer)
            else:
                # Fast mode: push transitions directly to replay buffer
                for transition in self.episode_buffer:
                    state, action, reward, next_state, done, _ = transition
                    self.replay_buffer.push(state, action, reward, next_state, done)
            self.episode_buffer = []

    def update(self, batch_size=256):
        """
        Update the networks using a batch from the replay buffer.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of losses
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors with validation
        rgb_array = np.array([s['rgb'] for s in states])
        depth_array = np.array([s['depth'] for s in states])

        # Replace NaN/Inf with valid values
        rgb_array = np.nan_to_num(rgb_array, nan=0.0, posinf=255.0, neginf=0.0)
        depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=10.0, neginf=0.0)

        rgb_batch = torch.FloatTensor(rgb_array).to(self.device)
        depth_batch = torch.FloatTensor(depth_array).to(self.device)
        desired_goal_batch = torch.FloatTensor(np.array([s['desired_goal'] for s in states])).to(self.device)
        achieved_goal_batch = torch.FloatTensor(np.array([s['achieved_goal'] for s in states])).to(self.device)

        next_rgb_array = np.array([s['rgb'] for s in next_states])
        next_depth_array = np.array([s['depth'] for s in next_states])

        # Replace NaN/Inf with valid values
        next_rgb_array = np.nan_to_num(next_rgb_array, nan=0.0, posinf=255.0, neginf=0.0)
        next_depth_array = np.nan_to_num(next_depth_array, nan=0.0, posinf=10.0, neginf=0.0)

        next_rgb_batch = torch.FloatTensor(next_rgb_array).to(self.device)
        next_depth_batch = torch.FloatTensor(next_depth_array).to(self.device)
        next_desired_goal_batch = torch.FloatTensor(
            np.array([s['desired_goal'] for s in next_states])
        ).to(self.device)
        next_achieved_goal_batch = torch.FloatTensor(
            np.array([s['achieved_goal'] for s in next_states])
        ).to(self.device)

        actions_tensor = torch.LongTensor(actions).to(self.device)  # (B, 2)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Get next action masks (if available)
        next_action_masks = None
        if 'action_mask' in next_states[0]:
            next_action_masks = torch.BoolTensor(
                np.array([s['action_mask'] for s in next_states])
            ).to(self.device)

        # Update Q-networks
        with torch.no_grad():
            # Get next actions and log probs from actor
            (next_pixel_actions, next_rotation_actions), next_log_probs = self.actor_network(
                next_rgb_batch, next_depth_batch,
                next_desired_goal_batch, next_achieved_goal_batch,
                deterministic=False,
                action_mask=next_action_masks
            )

            # Compute target Q-values
            next_actions_tuple = (next_pixel_actions, next_rotation_actions)
            target_q1 = self.target_q_network_1(
                next_rgb_batch, next_depth_batch,
                next_desired_goal_batch, next_achieved_goal_batch,
                next_actions_tuple
            )
            target_q2 = self.target_q_network_2(
                next_rgb_batch, next_depth_batch,
                next_desired_goal_batch, next_achieved_goal_batch,
                next_actions_tuple
            )
            target_q = torch.min(target_q1, target_q2)

            # SAC target: r + gamma * (min(Q1, Q2) - alpha * log_prob)
            alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
            target_q = rewards_tensor.unsqueeze(1) + (1 - dones_tensor.unsqueeze(1)) * self.gamma * (
                target_q - alpha * next_log_probs.unsqueeze(1)
            )

        # Current Q-values
        current_actions_tuple = (actions_tensor[:, 0], actions_tensor[:, 1])
        current_q1 = self.q_network_1(
            rgb_batch, depth_batch,
            desired_goal_batch, achieved_goal_batch,
            current_actions_tuple
        )
        current_q2 = self.q_network_2(
            rgb_batch, depth_batch,
            desired_goal_batch, achieved_goal_batch,
            current_actions_tuple
        )

        # Q-network losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update Q-networks
        self.q_optimizer_1.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=1.0)
        self.q_optimizer_1.step()

        self.q_optimizer_2.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network_2.parameters(), max_norm=1.0)
        self.q_optimizer_2.step()

        # Get action masks from batch (if available)
        action_masks = None
        if 'action_mask' in states[0]:
            action_masks = torch.BoolTensor(
                np.array([s['action_mask'] for s in states])
            ).to(self.device)

        # Update actor network
        (pixel_actions, rotation_actions), log_probs = self.actor_network(
            rgb_batch, depth_batch,
            desired_goal_batch, achieved_goal_batch,
            deterministic=False,
            action_mask=action_masks
        )

        actions_tuple = (pixel_actions, rotation_actions)
        q1_pi = self.q_network_1(
            rgb_batch, depth_batch,
            desired_goal_batch, achieved_goal_batch,
            actions_tuple
        )
        q2_pi = self.q_network_2(
            rgb_batch, depth_batch,
            desired_goal_batch, achieved_goal_batch,
            actions_tuple
        )
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss: maximize (Q - alpha * log_prob)
        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        actor_loss = (alpha * log_probs.unsqueeze(1) - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update alpha (temperature)
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.q_network_1, self.target_q_network_1)
        self._soft_update(self.q_network_2, self.target_q_network_2)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item() if self.auto_alpha else self.alpha,
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0
        }

    def _soft_update(self, source, target):
        """
        Soft update target network parameters.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def save(self, filepath):
        """
        Save model checkpoints.
        """
        torch.save({
            'q_network_1': self.q_network_1.state_dict(),
            'q_network_2': self.q_network_2.state_dict(),
            'target_q_network_1': self.target_q_network_1.state_dict(),
            'target_q_network_2': self.target_q_network_2.state_dict(),
            'actor_network': self.actor_network.state_dict(),
            'q_optimizer_1': self.q_optimizer_1.state_dict(),
            'q_optimizer_2': self.q_optimizer_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.log_alpha is not None else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None,
        }, filepath)

    def load(self, filepath, strict=True):
        """
        Load model checkpoints.

        Args:
            filepath: Path to checkpoint file
            strict: If True, require exact match. If False, load only compatible layers.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        partial_load_occurred = False

        if strict:
            # Try strict loading first
            try:
                self.q_network_1.load_state_dict(checkpoint['q_network_1'], strict=True)
                self.q_network_2.load_state_dict(checkpoint['q_network_2'], strict=True)
                self.target_q_network_1.load_state_dict(checkpoint['target_q_network_1'], strict=True)
                self.target_q_network_2.load_state_dict(checkpoint['target_q_network_2'], strict=True)
                self.actor_network.load_state_dict(checkpoint['actor_network'], strict=True)
            except RuntimeError as e:
                # If strict=True, re-raise the error
                raise e

        if not strict:
            # Partial loading: load only compatible layers (encoder and embeddings)
            # Skip FC layers that have dimension mismatches
            def load_partial_state_dict(model, checkpoint_dict, model_name=""):
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint_dict.items()
                                   if k in model_dict and model_dict[k].shape == v.shape}
                skipped = []
                for k in checkpoint_dict:
                    if k not in model_dict:
                        skipped.append(f"{k} (not in current model)")
                    elif model_dict[k].shape != checkpoint_dict[k].shape:
                        skipped.append(f"{k} (shape mismatch: {checkpoint_dict[k].shape} vs {model_dict[k].shape})")

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                return skipped

            # Load Q-networks with partial matching
            skipped_q1 = load_partial_state_dict(self.q_network_1, checkpoint['q_network_1'], "Q1")
            skipped_q2 = load_partial_state_dict(self.q_network_2, checkpoint['q_network_2'], "Q2")
            skipped_tq1 = load_partial_state_dict(
                self.target_q_network_1, checkpoint['target_q_network_1'], "Target Q1"
            )
            skipped_tq2 = load_partial_state_dict(
                self.target_q_network_2, checkpoint['target_q_network_2'], "Target Q2"
            )
            skipped_actor = load_partial_state_dict(self.actor_network, checkpoint['actor_network'], "Actor")

            # Log skipped layers (unique)
            all_skipped = set(skipped_q1 + skipped_q2 + skipped_tq1 + skipped_tq2 + skipped_actor)
            if all_skipped:
                partial_load_occurred = True
                import warnings
                warnings.warn(f"Partially loaded model. Skipped layers: {', '.join(sorted(all_skipped))}")

        # Load optimizers (may fail if architecture changed, that's okay)
        # If partial loading occurred, skip optimizer loading and reinitialize them
        if not partial_load_occurred:
            try:
                self.q_optimizer_1.load_state_dict(checkpoint['q_optimizer_1'])
                self.q_optimizer_2.load_state_dict(checkpoint['q_optimizer_2'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            except (KeyError, ValueError, RuntimeError):
                # Optimizer states may not match if architecture changed - that's okay
                # Reinitialize optimizers with current model parameters
                import warnings
                warnings.warn("Optimizer state could not be loaded. Reinitializing optimizers.")
                # Get learning rate from existing optimizer
                lr = self.q_optimizer_1.param_groups[0]['lr']
                self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr, weight_decay=0.0)
                self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr, weight_decay=0.0)
                self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr, weight_decay=0.0)
        else:
            # Partial load occurred - reinitialize optimizers since architecture changed
            import warnings
            warnings.warn("Partial model load detected. Reinitializing optimizers to match new architecture.")
            # Get learning rate from existing optimizer
            lr = self.q_optimizer_1.param_groups[0]['lr']
            self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=lr, weight_decay=0.0)
            self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=lr, weight_decay=0.0)
            self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr, weight_decay=0.0)

        # Load alpha if available
        if checkpoint.get('log_alpha') is not None and self.log_alpha is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
        if checkpoint.get('alpha_optimizer') is not None and self.alpha_optimizer is not None:
            try:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            except (KeyError, ValueError, RuntimeError):
                # Reinitialize alpha optimizer if loading fails
                lr = self.alpha_optimizer.param_groups[0]['lr'] if self.alpha_optimizer else 3e-4
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr, weight_decay=0.0)
