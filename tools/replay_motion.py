import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time

MODEL_PATH  = "models/g1/scene.xml"
MOTION_DIR  = "motions/moonwalk"
SIM_DT      = 0.002
MOTION_FPS  = 50
MOTION_DT   = 1.0 / MOTION_FPS

# Soma-retargeter joint order (matches _dof column order in raw CSV)
SOMA_JOINT_NAMES = [
    'left_hip_pitch_joint',       # joint_0
    'left_hip_roll_joint',        # joint_1
    'left_hip_yaw_joint',         # joint_2
    'left_knee_joint',            # joint_3
    'left_ankle_pitch_joint',     # joint_4
    'left_ankle_roll_joint',      # joint_5
    'right_hip_pitch_joint',      # joint_6
    'right_hip_roll_joint',       # joint_7
    'right_hip_yaw_joint',        # joint_8
    'right_knee_joint',           # joint_9
    'right_ankle_pitch_joint',    # joint_10
    'right_ankle_roll_joint',     # joint_11
    'waist_yaw_joint',            # joint_12
    'waist_roll_joint',           # joint_13
    'waist_pitch_joint',          # joint_14
    'left_shoulder_pitch_joint',  # joint_15
    'left_shoulder_roll_joint',   # joint_16
    'left_shoulder_yaw_joint',    # joint_17
    'left_elbow_joint',           # joint_18
    'left_wrist_roll_joint',      # joint_19
    'left_wrist_pitch_joint',     # joint_20
    'left_wrist_yaw_joint',       # joint_21
    'right_shoulder_pitch_joint', # joint_22
    'right_shoulder_roll_joint',  # joint_23
    'right_shoulder_yaw_joint',   # joint_24
    'right_elbow_joint',          # joint_25
    'right_wrist_roll_joint',     # joint_26
    'right_wrist_pitch_joint',    # joint_27
    'right_wrist_yaw_joint',      # joint_28
]

def build_soma_to_mj_map(model):
    mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(model.nu)]
    soma_to_mj = np.full(model.nu, -1, dtype=int)
    for mj_idx, mj_name in enumerate(mj_names):
        for soma_idx, soma_name in enumerate(SOMA_JOINT_NAMES):
            if soma_name in mj_name or mj_name in soma_name:
                soma_to_mj[mj_idx] = soma_idx
                break
    unmapped = np.where(soma_to_mj < 0)[0]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} unmapped actuators")
    else:
        print(f"All {model.nu} actuators mapped OK")
    return soma_to_mj

def set_pose(model, data, joint_pos, body_pos, frame_idx, z_offset, soma_to_mj):
    soma_joints = joint_pos[frame_idx]
    mj_joints   = np.zeros(model.nu, dtype=np.float32)
    for mj_idx in range(model.nu):
        soma_idx = soma_to_mj[mj_idx]
        if soma_idx >= 0:
            mj_joints[mj_idx] = soma_joints[soma_idx]

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        lo = model.jnt_range[joint_id, 0]
        hi = model.jnt_range[joint_id, 1]
        mj_joints[i] = np.clip(mj_joints[i], lo, hi)

    data.qpos[7:] = mj_joints
    data.qpos[0]  = body_pos[frame_idx, 0]
    data.qpos[1]  = body_pos[frame_idx, 1]
    data.qpos[2]  = body_pos[frame_idx, 2] - z_offset
    data.qpos[3]  = 1.0
    data.qpos[4]  = 0.0
    data.qpos[5]  = 0.0
    data.qpos[6]  = 0.0
    data.qvel[:]  = 0.0

    mujoco.mj_kinematics(model, data)
    mujoco.mj_fwdPosition(model, data)

def run():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model.opt.gravity[:] = 0.0

    soma_to_mj = build_soma_to_mj_map(model)

    joint_pos   = pd.read_csv(f"{MOTION_DIR}/joint_pos.csv").values.astype(np.float32)
    body_pos_df = pd.read_csv(f"{MOTION_DIR}/body_pos.csv")
    body_pos    = body_pos_df[['body_0_x','body_0_y','body_0_z']].values.astype(np.float32)

    n_frames = len(joint_pos)
    print(f"Loaded {n_frames} frames = {n_frames/MOTION_FPS:.1f}s")

    z_min    = body_pos[:, 2].min()
    z_offset = z_min - 0.75
    print(f"Z offset: {z_offset:.4f}m")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.trackbodyid = 1
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        print("Starting in 3 seconds...")
        set_pose(model, data, joint_pos, body_pos, 0, z_offset, soma_to_mj)
        for _ in range(int(3.0 / SIM_DT)):
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(SIM_DT)

        print("Replaying motion...")
        for frame_idx in range(n_frames):
            if not viewer.is_running():
                break
            set_pose(model, data, joint_pos, body_pos, frame_idx, z_offset, soma_to_mj)
            viewer.sync()
            time.sleep(MOTION_DT * 1.5)

        print("Done.")

    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        MOTION_DIR = sys.argv[1]
    run()
