"""
add_body_data.py
================
Builds SONIC reference folders with full body kinematics.

SONIC's encoder reads `vr_3point_local_target` (head and both hands) and
`motion_anchor_orientation` from the body arrays in a reference folder, so
those arrays need the true world pose of every tracked body, not just the
root.

WHAT THIS DOES
  Runs forward kinematics in MuJoCo, frame by frame, and records the world
  position and orientation of the 14 bodies SONIC tracks. Also:
    - reorders joints from SOMA/MuJoCo order to IsaacLab order (what the
      reference reader expects)
    - recomputes joint and body velocities by finite difference at 50 Hz
    - shifts the clip vertically so the lowest ankle sits on the floor

  Output folders are deploy-ready: copy them straight into
  gear_sonic_deploy/reference/<something>/

USAGE  (from the repo root)
    uv run python tools/add_body_data.py
    uv run python tools/add_body_data.py --motions wave walk
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

MOTION_FPS = 50.0

# SOMA/MuJoCo joint order -> IsaacLab order (what reference CSVs must use)
IL_TO_MJ = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
            5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

# The 14 bodies SONIC tracks, as declared in metadata.txt:
#   [0 4 10 18 5 11 19 9 16 22 28 17 23 29]
# Those are IsaacLab body indices; decoded against IsaacLab's breadth-first
# body ordering for the 29-DoF G1 they are, in order:
BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
ANKLE_SLOTS = [3, 6]          # indices into BODY_NAMES for the two ankles
ANKLE_FLOOR_Z = 0.03          # metres; ankle-roll link centre above the sole


def resolve_bodies(model):
    """Map BODY_NAMES to MuJoCo body ids, with helpful output on a miss."""
    ids, missing = [], []
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            missing.append(name)
        ids.append(bid)
    if missing:
        print("Could not find these bodies in the model:")
        for m in missing:
            print(f"  {m}")
        print("\nBodies the model does have:")
        for i in range(model.nbody):
            print(f"  [{i:2d}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")
        raise SystemExit(1)
    return ids


def quat_conj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def angular_velocity(quats, fps=MOTION_FPS):
    """Finite-difference angular velocity in the world frame, from wxyz quats."""
    n = len(quats)
    omega = np.zeros((n, quats.shape[1], 3), dtype=np.float32)
    if n < 2:
        return omega
    dq = quat_mul(quats[1:], quat_conj(quats[:-1]))
    # keep the short way round
    flip = dq[..., 0] < 0
    dq[flip] *= -1.0
    omega[1:] = 2.0 * dq[..., 1:] * fps
    omega[0] = omega[1]
    return omega


def process(src: Path, dst: Path, model, data, body_ids):
    joint_pos = pd.read_csv(src / "joint_pos.csv").values.astype(np.float64)
    bp = pd.read_csv(src / "body_pos.csv")
    bq = pd.read_csv(src / "body_quat.csv")
    root_pos = bp[["body_0_x", "body_0_y", "body_0_z"]].values.astype(np.float64)
    root_quat = bq[["body_0_w", "body_0_x", "body_0_y", "body_0_z"]].values.astype(np.float64)

    n = min(len(joint_pos), len(root_pos), len(root_quat))
    joint_pos, root_pos, root_quat = joint_pos[:n], root_pos[:n], root_quat[:n]
    if joint_pos.shape[1] != 29:
        raise ValueError(f"{src.name}: joint_pos has {joint_pos.shape[1]} cols, expected 29")

    nb = len(body_ids)
    pos = np.zeros((n, nb, 3), dtype=np.float32)
    quat = np.zeros((n, nb, 4), dtype=np.float32)

    # joint limits, so FK never sees an out-of-range pose
    lo = np.array([model.jnt_range[model.actuator_trnid[i, 0], 0] for i in range(29)])
    hi = np.array([model.jnt_range[model.actuator_trnid[i, 0], 1] for i in range(29)])

    for t in range(n):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]
        data.qpos[7:36] = np.clip(joint_pos[t], lo, hi)
        data.qvel[:] = 0.0
        mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(model, data)
        for k, bid in enumerate(body_ids):
            pos[t, k] = data.xpos[bid]
            quat[t, k] = data.xquat[bid]

    # put the lowest ankle on the floor (pure z translation, so FK still holds)
    ankle_z = pos[:, ANKLE_SLOTS, 2].min()
    shift = ANKLE_FLOOR_Z - ankle_z
    pos[:, :, 2] += shift

    lin_vel = np.zeros_like(pos)
    lin_vel[1:] = (pos[1:] - pos[:-1]) * MOTION_FPS
    lin_vel[0] = lin_vel[1]
    ang_vel = angular_velocity(quat)

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) * MOTION_FPS
    joint_vel[0] = joint_vel[1]
    joint_vel = np.clip(joint_vel, -20, 20)

    # SOMA/MuJoCo order -> IsaacLab order
    jp_il = joint_pos[:, IL_TO_MJ].astype(np.float32)
    jv_il = joint_vel[:, IL_TO_MJ].astype(np.float32)

    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(jp_il, columns=[f"joint_{i}" for i in range(29)]).to_csv(
        dst / "joint_pos.csv", index=False)
    pd.DataFrame(jv_il, columns=[f"joint_{i}" for i in range(29)]).to_csv(
        dst / "joint_vel.csv", index=False)
    pd.DataFrame(pos.reshape(n, -1),
                 columns=[f"body_{i // 3}_{'xyz'[i % 3]}" for i in range(nb * 3)]).to_csv(
        dst / "body_pos.csv", index=False)
    pd.DataFrame(quat.reshape(n, -1),
                 columns=[f"body_{i // 4}_{'wxyz'[i % 4]}" for i in range(nb * 4)]).to_csv(
        dst / "body_quat.csv", index=False)
    pd.DataFrame(lin_vel.reshape(n, -1),
                 columns=[f"body_{i // 3}_vel_{'xyz'[i % 3]}" for i in range(nb * 3)]).to_csv(
        dst / "body_lin_vel.csv", index=False)
    pd.DataFrame(ang_vel.reshape(n, -1),
                 columns=[f"body_{i // 3}_angvel_{'xyz'[i % 3]}" for i in range(nb * 3)]).to_csv(
        dst / "body_ang_vel.csv", index=False)

    idx = np.array(BODY_INDEXES)
    with open(dst / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {dst.name}\n{'=' * 30}\n\n")
        f.write(f"Body part indexes:\n{idx}\n\n")
        f.write(f"Total timesteps: {n}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: {jp_il.shape} (float32)\n")
        f.write(f"  joint_vel: {jv_il.shape} (float32)\n")
        f.write(f"  body_pos_w: {pos.shape} (float32)\n")
        f.write(f"  body_quat_w: {quat.shape} (float32)\n")
        f.write(f"  body_lin_vel_w: {lin_vel.shape} (float32)\n")
        f.write(f"  body_ang_vel_w: {ang_vel.shape} (float32)\n")
        f.write(f"  _body_indexes: {idx.shape} (int64)\n")
        f.write("  time_step_total: () (int64)\n")

    with open(dst / "info.txt", "w") as f:
        f.write(f"Motion Information: {dst.name}\n{'=' * 50}\n\n")
        f.write("joints: IsaacLab order (reordered from SOMA)\n")
        f.write(f"bodies: {nb} tracked, world frame, from MuJoCo forward kinematics\n")
        f.write(f"z shift applied: {shift:+.4f} m (lowest ankle -> {ANKLE_FLOOR_Z} m)\n\n")
        for name, arr in [("joint_pos", jp_il), ("joint_vel", jv_il),
                          ("body_pos_w", pos), ("body_quat_w", quat),
                          ("body_lin_vel_w", lin_vel), ("body_ang_vel_w", ang_vel)]:
            f.write(f"{name}:\n  Shape: {arr.shape}\n"
                    f"  Range: [{arr.min():.3f}, {arr.max():.3f}]\n\n")

    print(f"  {dst.name:<14} {n:4d} frames  z-shift {shift:+.3f}m  "
          f"hands z [{pos[:, [10, 13], 2].min():.2f}, {pos[:, [10, 13], 2].max():.2f}]m")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/g1/scene.xml")
    ap.add_argument("--src", default="motions", help="SOMA-ordered input folders")
    ap.add_argument("--out", default="motions_cmd", help="deploy-ready output folders")
    ap.add_argument("--motions", nargs="*",
                default=["walk", "moonwalk", "wave", "stretch"])
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}")
        return 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    body_ids = resolve_bodies(model)
    print(f"model ok: {model.nbody} bodies, {model.nu} actuators")
    print("tracking bodies resolved:")
    for name, bid in zip(BODY_NAMES, body_ids):
        print(f"  mj[{bid:2d}] {name}")
    print()

    src_root, out_root = Path(args.src), Path(args.out)
    for name in args.motions:
        src = src_root / name
        if not src.is_dir():
            print(f"  missing: {src}")
            continue
        process(src, out_root / f"{name}_il", model, data, body_ids)

    print(f"\nwrote {out_root}/ -- copy these into gear_sonic_deploy/reference/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
