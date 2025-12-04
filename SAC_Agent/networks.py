import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    """
    CNN encoder for processing RGB and depth images.
    """

    def __init__(self, rgb_shape=(200, 200, 3), depth_shape=(200, 200)):
        """
        Initialize CNN encoder.

        Args:
            rgb_shape: Shape of RGB image (H, W, C)
            depth_shape: Shape of depth image (H, W)
        """
        super(CNNEncoder, self).__init__()

        # RGB encoder
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Depth encoder (treat depth as single channel)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Flattened feature size: 64 * 8 * 8 = 4096 for each
        self.rgb_feature_size = 64 * 8 * 8
        self.depth_feature_size = 64 * 8 * 8

    def forward(self, rgb, depth):
        """
        Forward pass through encoders.

        Args:
            rgb: RGB image tensor (B, H, W, C) or (B, C, H, W)
            depth: Depth image tensor (B, H, W) or (B, 1, H, W)

        Returns:
            Combined feature vector
        """
        # Handle different input formats
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            rgb = rgb.permute(0, 3, 1, 2)
        if depth.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            depth = depth.unsqueeze(1)

        # Normalize RGB to [0, 1] and clamp to ensure valid range
        rgb = torch.clamp(rgb.float() / 255.0, min=0.0, max=1.0)

        # Encode RGB
        rgb_features = self.rgb_conv(rgb)
        rgb_features = rgb_features.reshape(rgb_features.size(0), -1)

        # Check for NaN/Inf in RGB features
        if torch.isnan(rgb_features).any() or torch.isinf(rgb_features).any():
            rgb_features = torch.nan_to_num(rgb_features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp depth to positive values (depth should be in meters, reasonable range 0-10m)
        depth = torch.clamp(depth.float(), min=0.0, max=10.0)

        # Encode depth
        depth_features = self.depth_conv(depth)
        depth_features = depth_features.reshape(depth_features.size(0), -1)

        # Check for NaN/Inf in depth features
        if torch.isnan(depth_features).any() or torch.isinf(depth_features).any():
            depth_features = torch.nan_to_num(depth_features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Concatenate features
        combined_features = torch.cat([rgb_features, depth_features], dim=1)

        # Final check on combined features
        if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
            combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=-1.0)

        return combined_features


class QNetwork(nn.Module):
    """
    Twin Q-network for SAC. Two separate Q-networks to reduce overestimation bias.
    """

    def __init__(self, rgb_shape=(200, 200, 3), depth_shape=(200, 200),
                 goal_dim=3, pixel_action_dim=40000, rotation_action_dim=6,
                 hidden_dim=256):
        """
        Initialize Q-network.

        Args:
            rgb_shape: Shape of RGB image
            depth_shape: Shape of depth image
            goal_dim: Dimension of goal (desired_goal and achieved_goal, each 3D)
            pixel_action_dim: Number of pixel positions (e.g., 200*200 = 40000)
            rotation_action_dim: Number of rotation options (e.g., 6)
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()

        # Image encoder
        self.encoder = CNNEncoder(rgb_shape, depth_shape)
        image_feature_dim = self.encoder.rgb_feature_size + self.encoder.depth_feature_size

        # Goal encoding (desired_goal and achieved_goal)
        # Handle goal_dim=0 for fast training mode
        self.goal_dim = goal_dim
        goal_input_dim = goal_dim * 2 if goal_dim > 0 else 0  # desired_goal + achieved_goal

        # Action encoding
        # For MultiDiscrete action: [pixel_position, rotation]
        # We'll use embedding for discrete actions
        self.pixel_embedding = nn.Embedding(pixel_action_dim, 64)
        self.rotation_embedding = nn.Embedding(rotation_action_dim, 16)
        action_embed_dim = 64 + 16

        # Combined input dimension
        combined_input_dim = image_feature_dim + goal_input_dim + action_embed_dim

        # Q-network layers
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, rgb, depth, desired_goal, achieved_goal, action):
        """
        Forward pass.

        Args:
            rgb: RGB image (B, H, W, C) or (B, C, H, W)
            depth: Depth image (B, H, W) or (B, 1, H, W)
            desired_goal: Desired goal (B, 3)
            achieved_goal: Achieved goal (B, 3)
            action: Action tuple (pixel_position, rotation) or tensor (B, 2)

        Returns:
            Q-value
        """
        # Encode images
        image_features = self.encoder(rgb, depth)

        # Encode goals (skip if goal_dim=0)
        if self.goal_dim > 0:
            goal_features = torch.cat([desired_goal, achieved_goal], dim=1)
        else:
            # Create empty tensor for goal features when goal_dim=0
            goal_features = torch.zeros(rgb.size(0), 0, device=rgb.device)

        # Encode actions
        if isinstance(action, tuple) or (isinstance(action, torch.Tensor) and action.dim() == 2):
            if isinstance(action, tuple):
                pixel_pos = action[0]
                rotation = action[1]
            else:
                pixel_pos = action[:, 0].long()
                rotation = action[:, 1].long()
        else:
            # Single action (not batched)
            pixel_pos = action[0].long()
            rotation = action[1].long()
            pixel_pos = pixel_pos.unsqueeze(0)
            rotation = rotation.unsqueeze(0)

        pixel_embed = self.pixel_embedding(pixel_pos)
        rotation_embed = self.rotation_embedding(rotation)
        action_features = torch.cat([pixel_embed, rotation_embed], dim=1)

        # Combine all features
        combined = torch.cat([image_features, goal_features, action_features], dim=1)

        #  Q-value
        q_value = self.fc_layers(combined)

        return q_value


class ActorNetwork(nn.Module):
    """
    Actor network for SAC that outputs action distributions.
    """

    def __init__(self, rgb_shape=(200, 200, 3), depth_shape=(200, 200),
                 goal_dim=3, pixel_action_dim=40000, rotation_action_dim=6,
                 hidden_dim=256):
        """
        Initialize actor network.

        Args:
            rgb_shape: Shape of RGB image
            depth_shape: Shape of depth image
            goal_dim: Dimension of goal
            pixel_action_dim: Number of pixel positions (200*200 = 40000)
            rotation_action_dim: Number of rotation options (6)
            hidden_dim: Hidden layer dimension
        """
        super(ActorNetwork, self).__init__()

        self.pixel_action_dim = pixel_action_dim
        self.rotation_action_dim = rotation_action_dim

        # Image encoder
        self.encoder = CNNEncoder(rgb_shape, depth_shape)
        image_feature_dim = self.encoder.rgb_feature_size + self.encoder.depth_feature_size

        # Goal encoding
        # Handle goal_dim=0 for fast training mode
        self.goal_dim = goal_dim
        goal_input_dim = goal_dim * 2 if goal_dim > 0 else 0  # desired_goal + achieved_goal

        # Combined input dimension
        combined_input_dim = image_feature_dim + goal_input_dim

        # Actor layers
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Output heads for discrete actions
        self.pixel_head = nn.Linear(hidden_dim, pixel_action_dim)
        self.rotation_head = nn.Linear(hidden_dim, rotation_action_dim)

        # Initialize weights properly to prevent NaNs
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights to prevent NaNs."""
        # Initialize all layers with very small weights to prevent numerical instability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use very small uniform initialization to prevent large values
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Use small uniform initialization for conv layers
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Initialize output heads with even smaller weights
        nn.init.uniform_(self.pixel_head.weight, -0.001, 0.001)
        nn.init.constant_(self.pixel_head.bias, 0.0)
        nn.init.uniform_(self.rotation_head.weight, -0.001, 0.001)
        nn.init.constant_(self.rotation_head.bias, 0.0)

        # Verify no NaNs in weights after initialization
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"WARNING: NaN/Inf found in {name} after initialization, fixing...")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.01, neginf=-0.01)

    def forward(self, rgb, depth, desired_goal, achieved_goal, deterministic=False, action_mask=None):
        """
        Forward pass.

        Args:
            rgb: RGB image
            depth: Depth image
            desired_goal: Desired goal
            achieved_goal: Achieved goal
            deterministic: If True, return deterministic action; else sample from distribution
            action_mask: Boolean mask for reachable pixels (H*W,) or (B, H*W). None to disable masking.

        Returns:
            Action tuple (pixel_position, rotation) and log probabilities
        """
        # Input validation: check for NaN/Inf
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            depth = torch.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)

        # Encode images
        image_features = self.encoder(rgb, depth)

        # Check for NaN in image features
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Encode goals (skip if goal_dim=0)
        if self.goal_dim > 0:
            goal_features = torch.cat([desired_goal, achieved_goal], dim=1)
            # Check for NaN in goal features
            if torch.isnan(goal_features).any() or torch.isinf(goal_features).any():
                goal_features = torch.nan_to_num(goal_features, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            # Create empty tensor for goal features when goal_dim=0
            goal_features = torch.zeros(rgb.size(0), 0, device=rgb.device)

        # Combine features
        combined = torch.cat([image_features, goal_features], dim=1)

        # Forward through layers
        hidden = self.fc_layers(combined)

        # Check for NaN in hidden features
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)

        # Get logits for each action dimension
        # Check hidden before computing logits
        if torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print("WARNING: NaN/Inf in hidden before logits computation")
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)

        pixel_logits = self.pixel_head(hidden)
        rotation_logits = self.rotation_head(hidden)

        # Immediately check and fix NaN/Inf in logits - this is critical
        if torch.isnan(pixel_logits).any() or torch.isinf(pixel_logits).any():
            print("WARNING: NaN/Inf detected in pixel_logits immediately after computation")
            nan_count = torch.isnan(pixel_logits).sum().item()
            inf_count = torch.isinf(pixel_logits).sum().item()
            print(f"  NaN count: {nan_count}, Inf count: {inf_count}")

            min_hidden = hidden.min().item()
            max_hidden = hidden.max().item()
            mean_hidden = hidden.mean().item()
            print(f"  Hidden stats: min={min_hidden:.6f}, max={max_hidden:.6f}, mean={mean_hidden:.6f}")

            min_weight = self.pixel_head.weight.min().item()
            max_weight = self.pixel_head.weight.max().item()
            print(f"  Pixel head weight stats: min={min_weight:.6f}, max={max_weight:.6f}")
            pixel_logits = torch.nan_to_num(pixel_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        if torch.isnan(rotation_logits).any() or torch.isinf(rotation_logits).any():
            print("WARNING: NaN/Inf detected in rotation_logits immediately after computation")
            rotation_logits = torch.nan_to_num(rotation_logits, nan=0.0, posinf=50.0, neginf=-50.0)

        # Clip logits to prevent NaN/Inf
        pixel_logits = torch.clamp(pixel_logits, min=-50.0, max=50.0)
        rotation_logits = torch.clamp(rotation_logits, min=-50.0, max=50.0)

        # Apply action mask to pixel logits if provided
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(pixel_logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)  # Add batch dimension
            # Mask unreachable pixels by setting logits to very negative value
            pixel_logits = pixel_logits.masked_fill(~action_mask.bool(), float('-inf'))

        # Final safety check before creating distributions
        if torch.isnan(pixel_logits).any() or torch.isinf(pixel_logits).any():
            # Replace NaN with zeros, keep -inf for masking
            pixel_logits = torch.where(torch.isnan(pixel_logits), torch.zeros_like(pixel_logits), pixel_logits)
            pixel_logits = torch.where(
                torch.isinf(pixel_logits) & (pixel_logits > 0), torch.ones_like(pixel_logits) * 50.0, pixel_logits
            )
        if torch.isnan(rotation_logits).any() or torch.isinf(rotation_logits).any():
            rotation_logits = torch.nan_to_num(rotation_logits, nan=0.0, posinf=50.0, neginf=-50.0)
            rotation_logits = torch.clamp(rotation_logits, min=-50.0, max=50.0)

        if deterministic:
            pixel_action = torch.argmax(pixel_logits, dim=1)
            rotation_action = torch.argmax(rotation_logits, dim=1)
            pixel_log_prob = None
            rotation_log_prob = None
        else:
            # Final check before creating distributions - replace any remaining NaNs
            # Use a more aggressive approach: replace all NaNs and ensure valid values
            pixel_logits = torch.where(
                torch.isnan(pixel_logits) | torch.isinf(pixel_logits),
                torch.zeros_like(pixel_logits),
                pixel_logits
            )
            # Ensure pixel_logits are finite and in valid range
            pixel_logits = torch.clamp(pixel_logits, min=-50.0, max=50.0)

            rotation_logits = torch.where(
                torch.isnan(rotation_logits) | torch.isinf(rotation_logits),
                torch.zeros_like(rotation_logits),
                rotation_logits
            )
            # Ensure rotation_logits are finite and in valid range
            rotation_logits = torch.clamp(rotation_logits, min=-50.0, max=50.0)

            # One more absolute safety check - if still NaN, use uniform distribution
            if torch.isnan(pixel_logits).any() or not torch.isfinite(pixel_logits).all():
                print("WARNING: NaN detected in pixel_logits, using uniform distribution")
                pixel_logits = torch.zeros_like(pixel_logits)
            if torch.isnan(rotation_logits).any() or not torch.isfinite(rotation_logits).all():
                print("WARNING: NaN detected in rotation_logits, using uniform distribution")
                rotation_logits = torch.zeros_like(rotation_logits)

            # Sample from categorical distributions
            pixel_dist = torch.distributions.Categorical(logits=pixel_logits)
            rotation_dist = torch.distributions.Categorical(logits=rotation_logits)

            pixel_action = pixel_dist.sample()
            rotation_action = rotation_dist.sample()

            pixel_log_prob = pixel_dist.log_prob(pixel_action)
            rotation_log_prob = rotation_dist.log_prob(rotation_action)

        # Combine log probabilities
        log_prob = pixel_log_prob + rotation_log_prob if pixel_log_prob is not None else None

        return (pixel_action, rotation_action), log_prob

    def get_action(self, rgb, depth, desired_goal, achieved_goal, deterministic=False, action_mask=None):
        """
        Get action for a single observation (no batch dimension).

        Args:
            rgb: RGB image
            depth: Depth image
            desired_goal: Desired goal
            achieved_goal: Achieved goal
            deterministic: If True, return deterministic action
            action_mask: Boolean mask for reachable pixels (H*W,). None to disable masking.
        """
        # Add batch dimension
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        if desired_goal.dim() == 1:
            desired_goal = desired_goal.unsqueeze(0)
        if achieved_goal.dim() == 1:
            achieved_goal = achieved_goal.unsqueeze(0)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(rgb.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            (pixel_action, rotation_action), _ = self.forward(
                rgb, depth, desired_goal, achieved_goal, deterministic, action_mask
            )

        # Remove batch dimension
        return (pixel_action.item(), rotation_action.item())
