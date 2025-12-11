"""
Observation preprocessing for DQN discrete model.

Based on the milestone document specification:
- Clamps depth above the table plane
- Injects small Gaussian depth noise
- Normalizes depth into [0, 1]
- Applies color jitter plus tensor conversion to RGB
"""

import numpy as np
import torch
import torchvision.transforms as transforms


def transform_observation(
    rgb,
    depth,
    table_height=0.91,
    depth_noise_std=0.01,
    apply_color_jitter=True,
    color_jitter_brightness=0.2,
    color_jitter_contrast=0.2,
    color_jitter_saturation=0.2,
    color_jitter_hue=0.1,
    training=True,
):
    """
    Transform observation as described in the milestone document.

    Args:
        rgb: RGB image (numpy array, shape: H, W, 3, dtype: uint8)
        depth: Depth image (numpy array, shape: H, W, dtype: float32, in meters)
        table_height: Height of table surface in meters (default: 0.91)
        depth_noise_std: Standard deviation for Gaussian depth noise (default: 0.01)
        apply_color_jitter: Whether to apply color jitter (default: True)
        color_jitter_brightness: Brightness jitter factor (default: 0.2)
        color_jitter_contrast: Contrast jitter factor (default: 0.2)
        color_jitter_saturation: Saturation jitter factor (default: 0.2)
        color_jitter_hue: Hue jitter factor (default: 0.1)
        training: If False, skip stochastic augmentations (default: True)

    Returns:
        Tuple of (rgb_tensor, depth_tensor) where:
        - rgb_tensor: RGB image tensor (shape: H, W, 3, dtype: float32, range: [0, 1])
        - depth_tensor: Depth image tensor (shape: H, W, dtype: float32, range: [0, 1])
    """
    # Ensure inputs are numpy arrays
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    # Handle RGB: ensure uint8 format
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Handle depth: ensure float32 format
    depth = depth.astype(np.float32)

    # 1. Clamp depth above the table plane
    # Only keep depth values that are at or above the table height
    depth = np.clip(depth, table_height, depth.max() if depth.max() > table_height else 10.0)

    # 2. Inject small Gaussian depth noise (only during training)
    if training and depth_noise_std > 0:
        depth_noise = np.random.normal(0, depth_noise_std, depth.shape).astype(np.float32)
        depth = depth + depth_noise
        # Re-clamp after noise injection
        depth = np.clip(depth, table_height, 10.0)

    # 3. Normalize depth into [0, 1]
    # Assuming depth range is [table_height, 10.0] meters
    depth_max = 10.0
    depth = (depth - table_height) / (depth_max - table_height)
    depth = np.clip(depth, 0.0, 1.0)

    # 4. Apply color jitter to RGB (only during training)
    if training and apply_color_jitter:
        # Convert RGB to PIL Image for color jitter
        from PIL import Image
        rgb_pil = Image.fromarray(rgb)

        # Create color jitter transform
        color_jitter = transforms.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )
        rgb_pil = color_jitter(rgb_pil)
        rgb = np.array(rgb_pil)

    # 5. Convert RGB to tensor and normalize to [0, 1]
    rgb_tensor = torch.from_numpy(rgb).float() / 255.0
    rgb_tensor = torch.clamp(rgb_tensor, min=0.0, max=1.0)

    # 6. Convert depth to tensor
    depth_tensor = torch.from_numpy(depth).float()
    depth_tensor = torch.clamp(depth_tensor, min=0.0, max=1.0)

    return rgb_tensor, depth_tensor


def transform_observation_batch(
    rgb_batch,
    depth_batch,
    table_height=0.91,
    depth_noise_std=0.01,
    apply_color_jitter=True,
    training=True,
):
    """
    Transform a batch of observations.

    Args:
        rgb_batch: List or array of RGB images
        depth_batch: List or array of depth images
        table_height: Height of table surface in meters
        depth_noise_std: Standard deviation for Gaussian depth noise
        apply_color_jitter: Whether to apply color jitter
        training: If False, skip stochastic augmentations

    Returns:
        Tuple of (rgb_tensors, depth_tensors) as torch tensors with batch dimension
    """
    rgb_tensors = []
    depth_tensors = []

    for rgb, depth in zip(rgb_batch, depth_batch):
        rgb_t, depth_t = transform_observation(
            rgb,
            depth,
            table_height=table_height,
            depth_noise_std=depth_noise_std,
            apply_color_jitter=apply_color_jitter,
            training=training,
        )
        rgb_tensors.append(rgb_t)
        depth_tensors.append(depth_t)

    # Stack into batch tensors
    rgb_batch_tensor = torch.stack(rgb_tensors)
    depth_batch_tensor = torch.stack(depth_tensors)

    return rgb_batch_tensor, depth_batch_tensor
