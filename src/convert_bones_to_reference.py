import numpy as np
import pandas as pd
import os
from scipy.spatial import transform

MJ_TO_IL = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28], dtype=np.int32)

def convert(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    T = len(df)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    motion_dir = os.path.join(out_dir, name)
    os.makedirs(motion_dir, exist_ok=True)

    # Root pos cm->m
    root_pos = df[['root_translateX','root_translateY','root_translateZ']].values.astype(np.float32) / 100.0

    # Root rotation degrees->quat wxyz
    euler = df[['root_rotateX','root_rotateY','root_rotateZ']].values
    quat_xyzw = transform.Rotation.from_euler('xyz', euler, degrees=True).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3,0,1,2]]

    # Joint pos degrees->radians, downsample 120->50fps
    joint_cols = [c for c in df.columns if c.endswith('_dof')]
    joint_pos = np.deg2rad(df[joint_cols].values).astype(np.float32)

    # Downsample 120fps -> 50fps
    step = 120/50
    idxs = np.arange(0, T, step).astype(int)
    idxs = idxs[idxs < T]
    joint_pos = joint_pos[idxs]
    root_pos = root_pos[idxs]
    quat_wxyz = quat_wxyz[idxs]
    T2 = len(idxs)

    # Joint vel from finite diff
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) * 50.0
    joint_vel[0] = joint_vel[1]
    joint_vel = np.clip(joint_vel, -20, 20)

    # body_pos: (T, 14, 3) - only root populated
    body_pos = np.zeros((T2, 14, 3), dtype=np.float32)
    body_pos[:, 0, :] = root_pos

    # body_quat: (T, 14, 4) wxyz - only root populated
    body_quat = np.zeros((T2, 14, 4), dtype=np.float32)
    body_quat[:, :, 0] = 1.0
    body_quat[:, 0, :] = quat_wxyz

    # body_lin_vel, body_ang_vel: zeros
    body_lin_vel = np.zeros((T2, 14, 3), dtype=np.float32)
    body_ang_vel = np.zeros((T2, 14, 3), dtype=np.float32)

    # Save CSVs
    pd.DataFrame(joint_pos, columns=[f'joint_{i}' for i in range(29)]).to_csv(f'{motion_dir}/joint_pos.csv', index=False)
    pd.DataFrame(joint_vel, columns=[f'joint_{i}' for i in range(29)]).to_csv(f'{motion_dir}/joint_vel.csv', index=False)
    pd.DataFrame(body_pos.reshape(T2,-1), columns=[f'body_{i//3}_{"xyz"[i%3]}' for i in range(42)]).to_csv(f'{motion_dir}/body_pos.csv', index=False)
    pd.DataFrame(body_quat.reshape(T2,-1), columns=[f'body_{i//4}_{"wxyz"[i%4]}' for i in range(56)]).to_csv(f'{motion_dir}/body_quat.csv', index=False)
    pd.DataFrame(body_lin_vel.reshape(T2,-1), columns=[f'body_{i//3}_vel_{"xyz"[i%3]}' for i in range(42)]).to_csv(f'{motion_dir}/body_lin_vel.csv', index=False)
    pd.DataFrame(body_ang_vel.reshape(T2,-1), columns=[f'body_{i//3}_angvel_{"xyz"[i%3]}' for i in range(42)]).to_csv(f'{motion_dir}/body_ang_vel.csv', index=False)

    # metadata.txt
    body_indexes = np.array([0,4,10,18,5,11,19,9,16,22,28,17,23,29])
    with open(f'{motion_dir}/metadata.txt','w') as f:
        f.write(f'Metadata for: {name}\n{"="*30}\n\n')
        f.write(f'Body part indexes:\n{body_indexes}\n\n')
        f.write(f'Total timesteps: {T2}\n\n')
        f.write('Data arrays summary:\n')
        f.write(f'  joint_pos: {joint_pos.shape} (float32)\n')
        f.write(f'  joint_vel: {joint_vel.shape} (float32)\n')
        f.write(f'  body_pos_w: {body_pos.shape} (float32)\n')
        f.write(f'  body_quat_w: {body_quat.shape} (float32)\n')
        f.write(f'  body_lin_vel_w: {body_lin_vel.shape} (float32)\n')
        f.write(f'  body_ang_vel_w: {body_ang_vel.shape} (float32)\n')
        f.write(f'  _body_indexes: {body_indexes.shape} (int64)\n')
        f.write(f'  time_step_total: () (int64)\n')

    # info.txt
    with open(f'{motion_dir}/info.txt','w') as f:
        f.write(f'Motion Information: {name}\n{"="*50}\n\n')
        f.write(f'joint_pos:\n  Shape: {joint_pos.shape}\n  Dtype: float32\n  Range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]\n\n')
        f.write(f'time_step_total:\n  Shape: ()\n  Value: {T2}\n')

    print(f'Done: {name} -> {T2} frames at 50fps')

if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1]
    out_dir = sys.argv[2]
    convert(csv_path, out_dir)
