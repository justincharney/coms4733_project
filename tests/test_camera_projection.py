import numpy as np
import pytest
import torch

from Grasping_Agent_multidiscrete import Grasp_Agent
from gym_grasper.controller.MujocoController import MJ_Controller


def test_world_pixel_roundtrip_top_down():
    controller = MJ_Controller(viewer=False)
    width, height = 200, 200
    camera = "top_down"
    controller.create_camera_data(width, height, camera)

    # Choose a point roughly over the table and in view of the top-down camera.
    world_point = np.array([0.0, -0.6, 1.0], dtype=np.float64)
    cam_point = controller.cam_rot_mat.T @ (world_point - controller.cam_pos)
    depth = -cam_point[2]
    assert depth > 0  # point should be in front of the camera

    pixel_x, pixel_y = controller.world_2_pixel(
        world_point, width=width, height=height, camera=camera
    )

    reconstructed = controller.pixel_2_world(
        pixel_x, pixel_y, depth, width=width, height=height, camera=camera
    )

    assert np.allclose(reconstructed, world_point, atol=1e-3)


def test_pixel_2_world_batch_matches_single():
    controller = MJ_Controller(viewer=False)
    width, height = 200, 200
    camera = "top_down"
    controller.create_camera_data(width, height, camera)

    # Sample a handful of pixels and plausible depths in front of the camera.
    pixel_x = np.array([0, 50, 100, 150, 199], dtype=np.int64)
    pixel_y = np.array([0, 60, 120, 180, 199], dtype=np.int64)
    depths = np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=np.float64)

    batch_world = controller.pixel_2_world_batch(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        depth=depths,
        width=width,
        height=height,
        camera=camera,
    )

    single_world = np.vstack(
        [
            controller.pixel_2_world(
                int(px), int(py), float(d), width=width, height=height, camera=camera
            )
            for px, py, d in zip(pixel_x, pixel_y, depths)
        ]
    )

    assert batch_world.shape == single_world.shape == (len(pixel_x), 3)
    assert np.allclose(batch_world, single_world, atol=1e-8)


def test_action_mask_batch_matches_loop():
    pytest.importorskip("mujoco")

    agent = Grasp_Agent(train=True)
    obs = agent.env.reset()

    # Batch path (uses pixel_2_world_batch)
    mask_batch = agent.compute_valid_action_mask(obs)

    # Force fallback to the per-pixel loop
    controller = agent.env.controller
    cls = controller.__class__
    had_batch = hasattr(cls, "pixel_2_world_batch")
    saved_batch = getattr(cls, "pixel_2_world_batch", None)
    try:
        if had_batch:
            delattr(cls, "pixel_2_world_batch")
        mask_loop = agent.compute_valid_action_mask(obs)
    finally:
        if had_batch:
            setattr(cls, "pixel_2_world_batch", saved_batch)
        agent.env.close()

    assert mask_batch.shape == mask_loop.shape
    assert torch.equal(mask_batch.cpu(), mask_loop.cpu())
