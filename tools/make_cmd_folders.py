"""
make_cmd_folders.py
===================
Reorders motions/<name>/ from SOMA/MuJoCo joint order into IsaacLab order and
writes motions_cmd/<name>_il/ -- the folders SONIC's reference reader loads.

Only joint_pos and joint_vel are permuted. Body arrays and the text files are
copied through unchanged.

USAGE  (from the repo root)
    uv run python tools/make_cmd_folders.py
    uv run python tools/make_cmd_folders.py --out motions_test --motions wave
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# SOMA/MuJoCo joint order -> IsaacLab order
IL_TO_MJ = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
            5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

PASSTHROUGH = ["body_pos.csv", "body_quat.csv", "body_lin_vel.csv",
               "body_ang_vel.csv", "metadata.txt", "info.txt"]


def convert(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    for fname in ("joint_pos.csv", "joint_vel.csv"):
        df = pd.read_csv(src / fname)
        arr = df.values.astype(np.float32)
        if arr.shape[1] != 29:
            raise ValueError(f"{src.name}/{fname}: {arr.shape[1]} cols, expected 29")
        pd.DataFrame(arr[:, IL_TO_MJ], columns=df.columns).to_csv(dst / fname, index=False)

    for fname in PASSTHROUGH:
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)

    n = len(pd.read_csv(dst / "joint_pos.csv"))
    print(f"  {dst.name:<14} {n:4d} frames  {len(list(dst.iterdir()))} files")


def main():
    ap = argparse.ArgumentParser(description="SOMA order -> IsaacLab order")
    ap.add_argument("--src", default="motions")
    ap.add_argument("--out", default="motions_cmd")
    ap.add_argument("--motions", nargs="*",
                    default=["walk", "moonwalk", "wave", "stretch"])
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    for name in args.motions:
        src = src_root / name
        if not src.is_dir():
            print(f"  missing: {src}")
            continue
        convert(src, out_root / f"{name}_il")

    print(f"\nwrote {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
