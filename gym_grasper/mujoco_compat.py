"""Minimal MuJoCo helpers that replace mujoco_py with the official mujoco package."""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
from gym import Env, error, spaces
from gym.utils import seeding

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict([(key, convert_observation_to_space(value)) for key, value in observation.items()])
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class ModelWrapper:
    """Light wrapper around mujoco.MjModel to provide mujoco_py-like helpers."""

    _JOINT_POS_DIMS = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }

    def __init__(self, model: mujoco.MjModel):
        object.__setattr__(self, "_raw_model", model)

    def __getattr__(self, item):
        return getattr(self._raw_model, item)

    def __setattr__(self, key, value):  # pragma: no cover - mirrors mujoco_py API
        setattr(self._raw_model, key, value)

    @property
    def raw(self) -> mujoco.MjModel:
        return self._raw_model

    @property
    def joint_names(self):
        return [mujoco.mj_id2name(self._raw_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self._raw_model.njnt)]

    def body_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._raw_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def joint_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._raw_model, mujoco.mjtObj.mjOBJ_JOINT, idx)

    def actuator_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._raw_model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)

    def camera_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._raw_model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def camera_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._raw_model, mujoco.mjtObj.mjOBJ_CAMERA, idx)

    def body_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._raw_model, mujoco.mjtObj.mjOBJ_BODY, idx)

    def get_joint_qpos_addr(self, joint_name: str) -> Tuple[int, int]:
        joint_id = mujoco.mj_name2id(self._raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        start = int(self._raw_model.jnt_qposadr[joint_id])
        joint_type = mujoco.mjtJoint(self._raw_model.jnt_type[joint_id])
        dim = self._JOINT_POS_DIMS.get(joint_type, 1)
        return start, start + dim


class SimWrapper:
    """Small wrapper mimicking mujoco_py.MjSim methods used in the project."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def render(
        self,
        width: int,
        height: int,
        camera_name: Optional[str] = None,
        camera_id: Optional[int] = None,
        depth: bool = False,
    ):
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer

        camera_kwargs = {}
        if camera_name is not None:
            camera_kwargs["camera"] = camera_name
        elif camera_id is not None:
            camera_kwargs["camera"] = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)

        renderer.update_scene(self.data, **camera_kwargs)
        if depth:
            rgb_img = renderer.render()
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, **camera_kwargs)
            depth_img = renderer.render()
            renderer.disable_depth_rendering()
            return np.asarray(rgb_img), np.asarray(depth_img)
        return np.asarray(renderer.render())


class MujocoEnv(Env):
    """Drop-in replacement for gym.envs.mujoco.MujocoEnv using mujoco>=3."""

    metadata = {"render.modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, model_path: str, frame_skip: int):
        if model_path.startswith(os.sep) or os.path.isabs(model_path):
            fullpath = model_path
        else:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            fullpath = os.path.join(assets_dir, model_path)
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        self.frame_skip = frame_skip
        raw_model = mujoco.MjModel.from_xml_path(fullpath)
        self.model = ModelWrapper(raw_model)
        self.sim = SimWrapper(raw_model)
        self.data = self.sim.data
        self._viewers: Dict[str, object] = {}

        self.metadata["video.frames_per_second"] = int(np.round(1.0 / self.dt))

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._set_action_space()
        action = self.action_space.sample()
        observation, _, done, _ = self.step(action)
        assert not done
        self._set_observation_space(observation)
        self.seed()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def viewer_setup(self):  # pragma: no cover - optional for subclasses
        pass

    def reset(self):
        self.sim.reset()
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model.raw, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array":
            return self.sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
                camera_id=camera_id,
                depth=False,
            )
        if mode == "depth_array":
            _, depth_img = self.sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
                camera_id=camera_id,
                depth=True,
            )
            return depth_img
        elif mode == "human":
            # There is no lightweight GUI viewer yet; fall back to rgb rendering.
            self.sim.render(width=width, height=height, camera_name=camera_name)
        else:  # pragma: no cover - handled by gym.Renderer usually
            raise error.Error(f"Unsupported render mode {mode}")

    def close(self):
        self._viewers = {}

    def get_body_com(self, body_name):
        return self.data.body_xpos[self.model.body_name2id(body_name)]

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
