"""
sonic_motion_publisher.py
=========================
Publishes a SOMA-retargeted motion to SONIC over ZMQ.

Wire format follows the reference publisher in the SONIC deploy stack
(gear_sonic_deploy/.../tests/test_zmq_manager.py):

  Message = topic_bytes + 1280-byte zero-padded JSON header + packed LE binary.

Topics used:
  command : start / stop / planner(bool) / optional delta_heading
  pose    : joint_pos[N,29], joint_vel[N,29], body_quat[N,4], frame_index[N], catch_up

The publisher BINDS; the C++ deploy CONNECTS to it (--zmq-host/--zmq-port).

JOINT ORDER
  The CSVs under motions/ are in SOMA/MuJoCo order (grouped by limb: legs
  0-11, waist 12-14, left arm 15-21, right arm 22-28). SONIC expects
  IsaacLab order, which interleaves by tree depth. load_motion() applies the
  IL_TO_MJ permutation on the way out -- without it, right_shoulder_pitch
  lands in the right_elbow slot.

USAGE
  # list configured motions
  python sonic_motion_publisher.py --list

  # stream one motion
  python sonic_motion_publisher.py --motion wave

  # load + validate the CSVs without sending anything
  python sonic_motion_publisher.py --motion wave --dry-run
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import zmq
except ImportError:
    zmq = None

HEADER_SIZE = 1280      # must match ZMQPackedMessageSubscriber::HEADER_SIZE
N_JOINTS = 29

# SOMA/MuJoCo joint order -> IsaacLab order. The SONIC policy expects
# IsaacLab ordering on both the reference CSVs and the ZMQ pose topic.
IL_TO_MJ = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,
            18,25,19,26,20,27,21,28]
MOTION_FPS = 50         # SONIC control rate
CHUNK_FRAMES = 50       # 1 second per chunk

# ── motion library ────────────────────────────────────────────────────────────

MOTION_ROOT = Path(__file__).resolve().parent.parent / "motions"


MOTIONS = {
    "walk":     MOTION_ROOT / "walk",
    "moonwalk": MOTION_ROOT / "moonwalk",
    "wave":     MOTION_ROOT / "wave",
    "stretch":  MOTION_ROOT / "stretch",
}

# ── loading ───────────────────────────────────────────────────────────────────
def load_motion(motion_dir: Path):
    """Load a SOMA reference folder into the arrays SONIC's pose topic wants."""
    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"motion dir not found: {motion_dir}")

    jp = pd.read_csv(motion_dir / "joint_pos.csv").values.astype(np.float32)
    jv = pd.read_csv(motion_dir / "joint_vel.csv").values.astype(np.float32)
    bq_df = pd.read_csv(motion_dir / "body_quat.csv")

    # root body is body_0; columns are body_0_w/x/y/z
    quat_cols = ["body_0_w", "body_0_x", "body_0_y", "body_0_z"]
    missing = [c for c in quat_cols if c not in bq_df.columns]
    if missing:
        raise ValueError(
            f"{motion_dir.name}/body_quat.csv missing {missing}. "
            f"Found: {list(bq_df.columns[:8])}"
        )
    bq = bq_df[quat_cols].values.astype(np.float32)

    n = min(len(jp), len(jv), len(bq))
    jp, jv, bq = jp[:n], jv[:n], bq[:n]

    if jp.shape[1] != N_JOINTS:
        raise ValueError(f"joint_pos has {jp.shape[1]} cols, expected {N_JOINTS}")
    if jv.shape[1] != N_JOINTS:
        raise ValueError(f"joint_vel has {jv.shape[1]} cols, expected {N_JOINTS}")

    jp = jp[:, IL_TO_MJ]
    jv = jv[:, IL_TO_MJ]

    return jp, jv, bq


def sanity_check(name, jp, jv, bq):
    """Cheap guardrail before anything reaches the controller."""
    print(f"  frames        : {len(jp)}  ({len(jp)/MOTION_FPS:.1f}s @ {MOTION_FPS}Hz)")
    print(f"  joint_pos     : {jp.shape}  range [{jp.min():+.3f}, {jp.max():+.3f}] rad")
    print(f"  joint_vel     : {jv.shape}  range [{jv.min():+.3f}, {jv.max():+.3f}] rad/s")
    print(f"  body_quat     : {bq.shape}  |q| mean {np.linalg.norm(bq, axis=1).mean():.4f}")

    problems = []
    if not np.isfinite(jp).all() or not np.isfinite(jv).all():
        problems.append("non-finite values in joint arrays")
    if np.abs(jp).max() > 4.0:
        problems.append(f"joint_pos max |{np.abs(jp).max():.2f}| rad looks out of range")
    qn = np.linalg.norm(bq, axis=1)
    if np.abs(qn - 1.0).max() > 0.05:
        problems.append(f"body_quat not unit norm (max dev {np.abs(qn-1.0).max():.3f})")

    for p in problems:
        print(f"  WARNING: {p}")
    return not problems


# ── ZMQ publisher ─────────────────────────────────────────────────────────────
class SonicPublisher:
    def __init__(self, host="*", port=5556, verbose=True):
        if zmq is None:
            raise RuntimeError("pyzmq not installed -- pip install pyzmq")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        endpoint = f"tcp://{host}:{port}"
        self.sock.bind(endpoint)
        self.verbose = verbose
        time.sleep(0.5)          # let subscribers connect before first send
        if verbose:
            print(f"[pub] bound {endpoint}, topics: command, pose")

    @staticmethod
    def _pack(topic: bytes, header: dict, data: bytes) -> bytes:
        hj = json.dumps(header).encode("utf-8")
        if len(hj) > HEADER_SIZE:
            raise ValueError(f"header {len(hj)}B exceeds {HEADER_SIZE}B")
        return topic + hj + b"\x00" * (HEADER_SIZE - len(hj)) + data

    def send_command(self, start, stop, planner, delta_heading=None):
        fields = [
            {"name": "start",   "dtype": "u8", "shape": [1]},
            {"name": "stop",    "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        data = struct.pack("BBB", int(start), int(stop), int(planner))
        if delta_heading is not None:
            fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
            data += struct.pack("<f", float(delta_heading))

        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        self.sock.send(self._pack(b"command", header, data))
        if self.verbose:
            print(f"[cmd] start={start} stop={stop} planner={planner}")

    def send_pose(self, joint_pos, joint_vel, body_quat, frame_indices, catch_up=False):
        n, j = joint_pos.shape
        header = {
            "v": 1, "endian": "le", "count": n,
            "fields": [
                {"name": "joint_pos",   "dtype": "f32", "shape": [n, j]},
                {"name": "joint_vel",   "dtype": "f32", "shape": [n, j]},
                {"name": "body_quat_w", "dtype": "f32", "shape": [n, 4]},
                {"name": "frame_index", "dtype": "i64", "shape": [n]},
                {"name": "catch_up",    "dtype": "u8",  "shape": [1]},
            ],
        }
        data = (
            joint_pos.astype(np.float32).tobytes()
            + joint_vel.astype(np.float32).tobytes()
            + body_quat.astype(np.float32).tobytes()
            + frame_indices.astype(np.int64).tobytes()
            + struct.pack("B", 1 if catch_up else 0)
        )
        self.sock.send(self._pack(b"pose", header, data))
        if self.verbose:
            print(f"[pose] {n} frames, idx {frame_indices[0]}..{frame_indices[-1]}")

    def close(self):
        self.sock.close()
        self.ctx.term()


# ── streaming ─────────────────────────────────────────────────────────────────
def stream_motion(pub, jp, jv, bq, start_frame=0, hold_sec=1.0):
    """Stream a motion in 1s chunks, real-time. Returns next free frame index.

    frame_index must stay monotonic across chunks and across motions,
    otherwise the controller treats it as a discontinuity.
    """
    n = len(jp)
    idx = start_frame

    for lo in range(0, n, CHUNK_FRAMES):
        hi = min(lo + CHUNK_FRAMES, n)
        count = hi - lo
        frames = np.arange(idx, idx + count, dtype=np.int64)
        pub.send_pose(jp[lo:hi], jv[lo:hi], bq[lo:hi], frames, catch_up=False)
        idx += count
        time.sleep(count / MOTION_FPS)      # real-time pacing

    # hold the final pose so the stream doesn't go silent mid-stance
    if hold_sec > 0:
        hold_n = int(hold_sec * MOTION_FPS)
        last_p = np.repeat(jp[-1:], hold_n, axis=0)
        last_v = np.zeros_like(last_p)
        last_q = np.repeat(bq[-1:], hold_n, axis=0)
        frames = np.arange(idx, idx + hold_n, dtype=np.int64)
        pub.send_pose(last_p, last_v, last_q, frames, catch_up=False)
        idx += hold_n
        time.sleep(hold_sec)

    return idx


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Stream a SOMA motion to SONIC over ZMQ")
    ap.add_argument("--motion", help="motion name from MOTIONS")
    ap.add_argument("--list", action="store_true", help="list configured motions")
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--dry-run", action="store_true", help="load + validate only, no ZMQ")
    ap.add_argument("--no-start", action="store_true",
                    help="skip the start command (robot already running)")
    args = ap.parse_args()

    if args.list or not args.motion:
        print("Configured motions:")
        for name, path in MOTIONS.items():
            mark = "ok " if path.is_dir() else "MISSING"
            print(f"  {mark}  {name:<12} {path}")
        if not args.motion:
            return 0

    if args.motion not in MOTIONS:
        print(f"unknown motion '{args.motion}'. known: {', '.join(MOTIONS)}")
        return 1

    print(f"\nLoading '{args.motion}' from {MOTIONS[args.motion]}")
    jp, jv, bq = load_motion(MOTIONS[args.motion])
    ok = sanity_check(args.motion, jp, jv, bq)

    if args.dry_run:
        print("\nDry run only -- nothing sent." if ok else "\nDry run: fix warnings first.")
        return 0 if ok else 1
    if not ok:
        print("\nRefusing to stream: sanity check failed.")
        return 1

    pub = SonicPublisher(host=args.host, port=args.port)
    try:
        if not args.no_start:
            # streamed-motion mode (planner=False) -- this is the ZMQ equivalent
            # of `]` start; the robot must already be stabilized (`9` in the sim).
            pub.send_command(start=True, stop=False, planner=False)
            time.sleep(0.5)

        print(f"\nStreaming {len(jp)} frames...")
        stream_motion(pub, jp, jv, bq, start_frame=0)
        print("Done.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
