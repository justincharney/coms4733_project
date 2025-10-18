"""
Basic MuJoCo viewer example.
Load and visualize a robot model with interactive controls.
"""

import mujoco
import mujoco.viewer
import os
import numpy as np


def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def main():
    # paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "scenes", "grasping_scene.xml")

    # load
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    model.opt.timestep = max(0.0015, model.opt.timestep)
    model.opt.iterations = max(80, model.opt.iterations)
    model.opt.ls_iterations = max(20, model.opt.ls_iterations)

    print(f"Grasping Environment loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    target_deg = [0.0, -30.0, 60.0, 0.0, 0.0, -15.0]  # safe “ready” bend
    base_targets = np.deg2rad(target_deg)

    # Gripper slightly open
    gripper_targets = np.array([0.01, -0.01], dtype=float)

    # Build actuator-sized target vector (truncate/extend as needed)
    target_positions = np.zeros(model.nu, dtype=float)
    fill = np.r_[base_targets, gripper_targets]
    target_positions[:min(model.nu, fill.size)] = fill[:min(model.nu, fill.size)]

    # Defaults: keep kd ~ 0.2–0.35 of kp for near-critical damping.
    kp_default = np.array([20, 20, 15, 8, 6, 6], dtype=float)
    kd_default = np.array([5,  5,  4,  3, 2, 2], dtype=float)
    # Extend with reasonable gripper gains
    kp = np.ones(model.nu, dtype=float) * (kp_default[-1] if kp_default.size else 10.0)
    kd = np.ones(model.nu, dtype=float) * (kd_default[-1] if kd_default.size else 2.0)
    n_copy = min(model.nu, kp_default.size)
    kp[:n_copy] = kp_default[:n_copy]
    kd[:n_copy] = kd_default[:n_copy]
    if model.nu >= 7:
        # make grippers stiff if they’re position/slide joints
        kp[6:] = 50.0
        kd[6:] = 5.0

    def actuator_joint_state(a_idx):
        """Return (jtype, q, qd, qadr, dadr) for actuator a_idx, or (None,)*5 if unsupported."""
        j_id = model.actuator_trnid[a_idx, 0]  # the joint this actuator drives
        if j_id < 0:
            return (None,)*5
        jtype = model.jnt_type[j_id]
        qadr = model.jnt_qposadr[j_id]
        dadr = model.jnt_dofadr[j_id]
        if jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            q = float(data.qpos[qadr])
            qd = float(data.qvel[dadr])
            return jtype, q, qd, qadr, dadr
        else:
            # Skip ball/free here
            return (None,)*5

    # Clip helper for actuator ctrlrange
    def clip_ctrl(a_idx, u):
        lo, hi = model.actuator_ctrlrange[a_idx]
        if lo < hi:
            return float(np.clip(u, lo, hi))
        return float(u)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Basic per-actuator PD (hinge/slide only), with angle wrapping for hinges
            for a in range(model.nu):
                jtype, q, qd, _, dadr = actuator_joint_state(a)
                if jtype is None:
                    # leaves them to passive dynamics
                    continue

                # position error (wrap for hinge)
                e = target_positions[a] - q
                if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                    e = wrap_to_pi(e)

                # velocity error (target velocity = 0)
                ed = -qd

                u = kp[a] * e + kd[a] * ed
                data.ctrl[a] = clip_ctrl(a, u)

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
