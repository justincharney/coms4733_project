import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    Basic ResNet residual block with two convolutional layers.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for processing RGB-D images.
    Based on the MULTIDISCRETE_RESNET architecture from the milestone document.
    """

    def __init__(self, rgb_shape=(200, 200, 3), depth_shape=(200, 200)):
        """
        Initialize ResNet encoder.

        Args:
            rgb_shape: Shape of RGB image (H, W, C)
            depth_shape: Shape of depth image (H, W)
        """
        super(ResNetEncoder, self).__init__()

        # RGB encoder - ResNet style
        self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_bn1 = nn.BatchNorm2d(64)
        self.rgb_relu = nn.ReLU(inplace=True)
        self.rgb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # RGB residual blocks
        self.rgb_layer1 = self._make_layer(64, 64, 2, stride=1)
        self.rgb_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.rgb_layer3 = self._make_layer(128, 256, 2, stride=2)

        # Depth encoder - ResNet style
        self.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_bn1 = nn.BatchNorm2d(64)
        self.depth_relu = nn.ReLU(inplace=True)
        self.depth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Depth residual blocks
        self.depth_layer1 = self._make_layer(64, 64, 2, stride=1)
        self.depth_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.depth_layer3 = self._make_layer(128, 256, 2, stride=2)

        # After 3 layers with stride 2 each, 200x200 -> 25x25
        # We'll use adaptive pooling to get consistent spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((25, 25))

        # Feature dimensions after encoding
        # Each branch outputs 256 channels at 25x25 = 256 * 25 * 25 = 160,000 features
        self.rgb_feature_size = 256 * 25 * 25
        self.depth_feature_size = 256 * 25 * 25

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        """
        Forward pass through ResNet encoders.

        Args:
            rgb: RGB image tensor (B, H, W, C) or (B, C, H, W)
            depth: Depth image tensor (B, H, W) or (B, 1, H, W)

        Returns:
            Combined feature map and flattened features
        """
        # Handle different input formats
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            rgb = rgb.permute(0, 3, 1, 2)
        if depth.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            depth = depth.unsqueeze(1)

        # RGB should already be normalized to [0, 1] by transform_observation
        # But handle both cases: if uint8 (0-255) or already normalized (0-1)
        if rgb.dtype == torch.uint8 or rgb.max() > 1.0:
            rgb = torch.clamp(rgb.float() / 255.0, min=0.0, max=1.0)
        else:
            rgb = torch.clamp(rgb.float(), min=0.0, max=1.0)

        # RGB ResNet forward pass
        rgb = self.rgb_conv1(rgb)
        rgb = self.rgb_bn1(rgb)
        rgb = self.rgb_relu(rgb)
        rgb = self.rgb_maxpool(rgb)
        rgb = self.rgb_layer1(rgb)
        rgb = self.rgb_layer2(rgb)
        rgb = self.rgb_layer3(rgb)
        rgb = self.adaptive_pool(rgb)  # (B, 256, 25, 25)

        # Depth should already be normalized to [0, 1] by transform_observation
        # But handle both cases: if in meters (0-10) or already normalized (0-1)
        if depth.max() > 1.0:
            # Still in meters, normalize
            depth = torch.clamp(depth.float(), min=0.0, max=10.0) / 10.0
        else:
            # Already normalized, just ensure it's in [0, 1]
            depth = torch.clamp(depth.float(), min=0.0, max=1.0)

        # Depth ResNet forward pass
        depth = self.depth_conv1(depth)
        depth = self.depth_bn1(depth)
        depth = self.depth_relu(depth)
        depth = self.depth_maxpool(depth)
        depth = self.depth_layer1(depth)
        depth = self.depth_layer2(depth)
        depth = self.depth_layer3(depth)
        depth = self.adaptive_pool(depth)  # (B, 256, 25, 25)

        # Concatenate RGB and depth features along channel dimension
        combined_features = torch.cat([rgb, depth], dim=1)  # (B, 512, 25, 25)

        # Flatten for fully connected layers
        combined_flat = combined_features.reshape(combined_features.size(0), -1)

        return combined_features, combined_flat


class DQNBaseNetwork(nn.Module):
    """
    DQN Base Network (MULTIDISCRETE_RESNET) as described in the milestone document.

    This network outputs Q-values for every combination of:
    - Pixel pick location: 200×200 = 40,000 positions
    - Gripper rotation: 6 angles (0°, ±30°, ±60°, 90°)
    - Total: 40,000 × 6 = 240,000 Q-values

    The architecture uses a ResNet-style encoder followed by spatial Q-value heads.
    """

    def __init__(self, rgb_shape=(200, 200, 3), depth_shape=(200, 200),
                 pixel_action_dim=40000, rotation_action_dim=6,
                 hidden_dim=512):
        """
        Initialize DQN base network.

        Args:
            rgb_shape: Shape of RGB image (H, W, C)
            depth_shape: Shape of depth image (H, W)
            pixel_action_dim: Number of pixel positions (200*200 = 40000)
            rotation_action_dim: Number of rotation options (6)
            hidden_dim: Hidden layer dimension for fully connected layers
        """
        super(DQNBaseNetwork, self).__init__()

        self.pixel_action_dim = pixel_action_dim
        self.rotation_action_dim = rotation_action_dim

        # ResNet encoder for RGB-D images
        self.encoder = ResNetEncoder(rgb_shape, depth_shape)

        # The encoder outputs features of shape (B, 512, 25, 25)
        # We need to upsample this to (B, C, 200, 200) to match pixel action space
        # Strategy: Use transposed convolutions to upsample from 25x25 to 200x200

        # Upsample from 25x25 to 200x200 (8x upsampling)
        # We'll do this in stages: 25 -> 50 -> 100 -> 200
        self.upsample = nn.Sequential(
            # 25x25 -> 50x50
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 100x100 -> 200x200
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final convolution to produce Q-values for each pixel-rotation combination
        # Output: (B, rotation_action_dim, 200, 200) = (B, 6, 200, 200)
        # This represents Q-values for each rotation at each pixel location
        self.q_head = nn.Conv2d(64, rotation_action_dim, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using standard ResNet initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize Q-head with small values
        nn.init.uniform_(self.q_head.weight, -0.01, 0.01)
        if self.q_head.bias is not None:
            nn.init.constant_(self.q_head.bias, 0)

    def forward(self, rgb, depth):
        """
        Forward pass.

        Args:
            rgb: RGB image (B, H, W, C) or (B, C, H, W)
            depth: Depth image (B, H, W) or (B, 1, H, W)

        Returns:
            Q-values tensor of shape (B, 200, 200, 6) or (B, 40000, 6)
            representing Q-values for each pixel-rotation combination
        """
        # Encode RGB-D images
        features_map, _ = self.encoder(rgb, depth)

        # Upsample features to match pixel action space (200x200)
        upsampled = self.upsample(features_map)  # (B, 64, 200, 200)

        # Generate Q-values for each rotation at each pixel
        q_values = self.q_head(upsampled)  # (B, 6, 200, 200)

        # Reshape to (B, 200, 200, 6) for easier indexing
        q_values = q_values.permute(0, 2, 3, 1)  # (B, 200, 200, 6)

        # Also provide flattened version: (B, 40000, 6)
        batch_size = q_values.size(0)
        q_values_flat = q_values.reshape(batch_size, self.pixel_action_dim, self.rotation_action_dim)

        return q_values, q_values_flat

    def get_q_value(self, rgb, depth, pixel_idx, rotation_idx):
        """
        Get Q-value for a specific pixel-rotation combination.

        Args:
            rgb: RGB image
            depth: Depth image
            pixel_idx: Pixel index (0-39999) or (row, col) tuple
            rotation_idx: Rotation index (0-5)

        Returns:
            Q-value for the specified action
        """
        q_values, _ = self.forward(rgb, depth)

        if isinstance(pixel_idx, tuple):
            row, col = pixel_idx
            return q_values[:, row, col, rotation_idx]
        else:
            # Flatten pixel index to (row, col)
            row = pixel_idx // 200
            col = pixel_idx % 200
            return q_values[:, row, col, rotation_idx]

    def get_best_action(self, rgb, depth, action_mask=None):
        """
        Get the best action (pixel, rotation) with highest Q-value.

        Args:
            rgb: RGB image
            depth: Depth image
            action_mask: Optional boolean mask (200, 200) for valid pixels

        Returns:
            Tuple of (pixel_idx, rotation_idx) with highest Q-value
        """
        q_values, _ = self.forward(rgb, depth)

        # Apply mask if provided
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(q_values.device)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0).unsqueeze(-1)  # (1, 200, 200, 1)
            # Mask invalid pixels by setting Q-values to very negative
            q_values = q_values.masked_fill(~action_mask.bool(), float('-inf'))

        # Find best action
        # Reshape to (B, 40000, 6) and find argmax
        batch_size = q_values.size(0)
        q_flat = q_values.reshape(batch_size, self.pixel_action_dim, self.rotation_action_dim)

        # Find best action for each batch item
        best_actions = []
        for b in range(batch_size):
            best_idx = torch.argmax(q_flat[b]).item()
            pixel_idx = best_idx // self.rotation_action_dim
            rotation_idx = best_idx % self.rotation_action_dim
            best_actions.append((pixel_idx, rotation_idx))

        if batch_size == 1:
            return best_actions[0]
        return best_actions
