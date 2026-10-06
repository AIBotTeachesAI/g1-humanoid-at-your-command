"""
zmq_session.py
==============
Persistent ZMQ session for SONIC's streamed-motion mode.

In streamed-motion mode the policy tracks whatever pose reference arrives on
the ZMQ `pose` topic, so the stream has to stay alive for the whole session,
not just while a motion is playing:
    idle       -> repeat a rest pose (frame 0 of a chosen motion)
    on command -> play that motion's frames
    after      -> hold its final pose

STARTUP ORDER
  The mode switch needs a live stream to attach to, so the stream starts
  first -- with catch_up=True for the first second, so the playback head
  syncs to the live edge -- and the mode is armed (planner on, planner off)
  only after that.

  frame_index stays monotonic across everything; a jump backwards reads as a
  discontinuity to the controller.

RUN IT ON THE SAME MACHINE AS THE DEPLOY (this binds, the deploy connects).

    source ~/GR00T-WholeBodyControl/.venv_sim/bin/activate
    cd ~/cmdlayer
    python src/zmq_session.py

  Then, in order:
    1. wait for "ready"
    2. press 9 in the MuJoCo window
    3. type a motion name at the prompt
    4. 'quit' to stop the policy and exit
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sonic_motion_publisher import (  # noqa: E402
    MOTION_FPS,
    MOTIONS,
    N_JOINTS,
    SonicPublisher,
    load_motion,
    sanity_check,
)

CHUNK = 5                       # frames per send -> 100 ms at 50 Hz
PERIOD = CHUNK / MOTION_FPS
WARMUP_CHUNKS = 10              # ~1 s of catch_up=True to sync the playback head


class Session:
    """Owns the socket. Only the streamer thread sends poses; the lock guards it."""

    def __init__(self, rest="wave", host="*", port=5556):
        self.pub = SonicPublisher(host=host, port=port, verbose=False)
        self.lock = threading.Lock()
        self.queue = deque()
        self.frame_idx = 0
        self.running = True
        self.warmup = WARMUP_CHUNKS
        self.cache = {}

        jp, jv, bq = self._load(rest)
        zero = np.zeros(N_JOINTS, dtype=np.float32)
        self.hold = (jp[0].copy(), zero, bq[0].copy())

        # 1. start control
        with self.lock:
            self.pub.send_command(start=True, stop=False, planner=False)
        print("start command sent")

        # 2. get pose data flowing BEFORE re-arming the mode
        self.thread = threading.Thread(target=self._stream, daemon=True)
        self.thread.start()
        time.sleep(WARMUP_CHUNKS * PERIOD + 0.2)
        print(f"streaming rest pose (frame 0 of '{rest}') at {MOTION_FPS} Hz")

        print("ready")

    def _load(self, name):
        if name not in self.cache:
            jp, jv, bq = load_motion(MOTIONS[name])
            if not sanity_check(name, jp, jv, bq):
                raise ValueError(f"{name} failed sanity check")
            self.cache[name] = (jp, jv, bq)
        return self.cache[name]

    def _stream(self):
        next_t = time.monotonic()
        zero = np.zeros(N_JOINTS, dtype=np.float32)
        while self.running:
            rows = []
            playing = False
            with self.lock:
                for _ in range(CHUNK):
                    if self.queue:
                        row = self.queue.popleft()
                        # hold the last played pose once the clip runs out
                        self.hold = (row[0], zero, row[2])
                        rows.append(row)
                        playing = True
                    else:
                        rows.append(self.hold)

            # Idle -> catch_up=True so the playback head rides the live edge
            # instead of stalling behind an ever-growing buffer. Playing ->
            # catch_up=False so the clip advances frame by frame.
            catch_up = not playing

            jp = np.stack([r[0] for r in rows]).astype(np.float32)
            jv = np.stack([r[1] for r in rows]).astype(np.float32)
            bq = np.stack([r[2] for r in rows]).astype(np.float32)
            idx = np.arange(self.frame_idx, self.frame_idx + CHUNK, dtype=np.int64)

            try:
                with self.lock:
                    self.pub.send_pose(jp, jv, bq, idx, catch_up=catch_up)
            except Exception as e:  # noqa: BLE001
                print(f"\nstream error: {type(e).__name__}: {e}")
                break

            self.frame_idx += CHUNK
            next_t += PERIOD
            time.sleep(max(0.0, next_t - time.monotonic()))

    def play(self, name, block=True):
        jp, jv, bq = self._load(name)
        n = len(jp)
        with self.lock:
            for i in range(n):
                self.queue.append((jp[i], jv[i], bq[i]))
        print(f"  queued {n} frames ({n / MOTION_FPS:.1f}s) from index {self.frame_idx}")
        if block:
            while True:
                with self.lock:
                    left = len(self.queue)
                if left == 0:
                    break
                time.sleep(0.05)
            time.sleep(0.3)
            print("  done, holding final pose")

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)
        try:
            with self.lock:
                self.pub.send_command(start=False, stop=True, planner=False)
            time.sleep(0.3)
        finally:
            self.pub.close()
        print("stopped")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Persistent SONIC ZMQ session")
    ap.add_argument("--rest", default="wave", help="motion whose frame 0 is the idle pose")
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--play", nargs="*", help="play these motions then exit")
    args = ap.parse_args()

    s = Session(rest=args.rest, host=args.host, port=args.port)
    try:
        if args.play:
            input("\npress 9 in the MuJoCo window to stabilize, then ENTER here: ")
            for name in args.play:
                print(f"\n> {name}")
                s.play(name)
        else:
            print("\nMotions:", ", ".join(MOTIONS))
            print("Press 9 in the MuJoCo window now. Then type a motion, or 'quit'.\n")
            while True:
                try:
                    cmd = input("motion> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not cmd:
                    continue
                if cmd.lower() in {"quit", "exit", "q"}:
                    break
                if cmd not in MOTIONS:
                    print(f"  unknown motion. known: {', '.join(MOTIONS)}")
                    continue
                s.play(cmd)
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
