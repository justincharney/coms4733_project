import numpy as np

from gym_grasper.controller.MujocoController import MJ_Controller


def test_world_pixel_roundtrip_top_down():
    controller = MJ_Controller(viewer=False)
    width, height = 200, 200
    camera = "top_down"
    controller.create_camera_data(width, height, camera)

    # Choose a point roughly over the table and in view of the top-down camera.
    world_point = np.array([0.0, -0.6, 1.0], dtype=np.float64)
    cam_point = controller.cam_rot_mat.T @ (world_point - controller.cam_pos)
    assert cam_point[2] > 0  # must be in front of the camera

    pixel_x, pixel_y = controller.world_2_pixel(
        world_point, width=width, height=height, camera=camera
    )
    depth = cam_point[2]

    reconstructed = controller.pixel_2_world(
        pixel_x, pixel_y, depth, width=width, height=height, camera=camera
    )

    assert np.allclose(reconstructed, world_point, atol=1e-3)
