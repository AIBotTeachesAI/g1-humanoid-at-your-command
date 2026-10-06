"""
keyboard_driver.py
==================
Drives SONIC's keyboard input interface programmatically, by running
`deploy.sh --input-type keyboard` under a pty and writing keystrokes to it.

This is the command channel: Nemotron chooses the motion, this selects it
and plays it on the robot.

KEY MAP (normal mode)
  ]  start control        T  play current motion to completion
  N  next motion          P  previous motion
  R  restart motion       O  stop control and exit
  9  stabilize -- NOT a deploy key. It goes to the MuJoCo window, so it
     stays manual: press it once after the policy starts.

MOTION SELECTION
  N does not print the motion name directly, but it sets reinitialize_heading,
  which makes the control loop print "Reference motion name: <name>". The
  driver presses N and reads that line, so it always knows what is selected
  without assuming any directory ordering.

PROMPT HANDLING
  deploy.sh's "Proceed with deployment? [Y/n]:" arrives without a trailing
  newline, so it only ever exists in the partial read buffer -- wait_for
  matches against that buffer as well as completed lines.

USAGE
    source ~/GR00T-WholeBodyControl/.venv_sim/bin/activate
    cd ~/cmdlayer
    python src/keyboard_driver.py                 # type motion names
    python src/keyboard_driver.py --llm           # type natural language
    python src/keyboard_driver.py --play wave_il  # one shot
"""

import argparse
import os
import pty
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

DEPLOY_DIR = Path.home() / "GR00T-WholeBodyControl" / "gear_sonic_deploy"
MOTION_DATA = "reference/"

RE_MOTION_NAME = re.compile(r"Reference motion name:\s*(\S+)")
RE_PLAYING = re.compile(r"Playing motion (\d+) from frame (\d+) to end \((\d+) total frames\)")
RE_COMPLETED = re.compile(r"Motion index:\s*(\d+)\s*:\s*(\S+)\s+completed")


class DeployDriver:
    def __init__(self, deploy_dir=DEPLOY_DIR, motion_data=MOTION_DATA, echo=False):
        self.deploy_dir = Path(deploy_dir)
        self.motion_data = motion_data
        self.echo = echo
        self.lines = deque(maxlen=4000)
        self.lock = threading.Lock()
        self.proc = None
        self.master_fd = None
        self.current_motion = None
        self._buf = ""

    # ── process ───────────────────────────────────────────────────────────
    def start(self, timeout=180):
        cmd = (
            f"source scripts/setup_env.sh && "
            f"export TensorRT_ROOT=/usr && "
            f"bash deploy.sh --input-type keyboard --motion-data {self.motion_data} sim"
        )
        self.master_fd, slave_fd = pty.openpty()
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(self.deploy_dir), close_fds=True, preexec_fn=os.setsid,
        )
        os.close(slave_fd)

        threading.Thread(target=self._reader, daemon=True).start()

        # deploy.sh asks "Proceed with deployment? [Y/n]"
        if self.wait_for(r"Proceed with deployment", timeout=60):
            self.send("y\n")

        print("waiting for Init Done (first run builds TensorRT engines, ~2 min)...")
        if not self.wait_for(r"Init Done", timeout=timeout):
            raise RuntimeError("deploy never reached 'Init Done' -- check the log")
        print("deploy ready")

    def _reader(self):
        while True:
            try:
                data = os.read(self.master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            with self.lock:
                self._buf += text
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.rstrip("\r")
                    self.lines.append(line)
                    m = RE_MOTION_NAME.search(line)
                    if m:
                        self.current_motion = m.group(1)
                    if self.echo:
                        print(f"  | {line}")

    # ── primitives ────────────────────────────────────────────────────────
    def send(self, keys):
        """Write raw keystrokes. No newline -- ENTER toggles planner mode."""
        os.write(self.master_fd, keys.encode())

    def wait_for(self, pattern, timeout=20.0, since=None):
        """Wait for a regex to appear in output. Returns the match or None."""
        rx = re.compile(pattern)
        start = time.monotonic()
        seen = 0 if since is None else since
        while time.monotonic() - start < timeout:
            with self.lock:
                lines = list(self.lines)
                partial = self._buf
            for line in lines[seen:]:
                m = rx.search(line)
                if m:
                    return m
            seen = len(lines)
            # prompts like "Proceed with deployment? [Y/n]:" arrive without a
            # trailing newline, so they only ever exist in the partial buffer
            if partial:
                m = rx.search(partial)
                if m:
                    return m
            time.sleep(0.05)
        return None

    def mark(self):
        with self.lock:
            return len(self.lines)

    # ── control ───────────────────────────────────────────────────────────
    def start_policy(self):
        self.send("]")
        time.sleep(0.5)
        print("policy started -- now press 9 in the MuJoCo window, let it settle ~5s")

    def list_motions(self, max_motions=12):
        """Cycle with N until a name repeats; returns the discovered order."""
        order = []
        for _ in range(max_motions):
            name = self.next_motion()
            if name is None:
                break
            if name in order:
                break
            order.append(name)
        return order

    def next_motion(self, timeout=5.0):
        at = self.mark()
        self.send("N")
        m = self.wait_for(r"Reference motion name:\s*(\S+)", timeout=timeout, since=at)
        return m.group(1) if m else None

    def select(self, name, max_presses=12):
        """Press N until `name` is the selected reference motion."""
        if self.current_motion == name:
            return True
        for _ in range(max_presses):
            got = self.next_motion()
            if got is None:
                print("  no response to N -- is the policy started?")
                return False
            if got == name:
                return True
        print(f"  '{name}' not found after {max_presses} presses")
        return False

    def play(self, timeout=60.0):
        at = self.mark()
        self.send("T")
        m = self.wait_for(RE_PLAYING.pattern, timeout=5.0, since=at)
        if not m:
            print("  no 'Playing motion' response")
            return False
        idx, frame, total = m.groups()
        print(f"  playing motion {idx}: {total} frames ({int(total) / 50:.1f}s)")
        done = self.wait_for(RE_COMPLETED.pattern, timeout=timeout, since=at)
        if done:
            print(f"  completed: {done.group(2)}")
            return True
        print("  did not report completion (robot may have fallen)")
        return False

    def stop(self):
        try:
            self.send("O")
            time.sleep(1.0)
        except Exception:  # noqa: BLE001
            pass
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        print("deploy stopped")

    def dump(self, n=40):
        with self.lock:
            for line in list(self.lines)[-n:]:
                print(f"  | {line}")


def main():
    ap = argparse.ArgumentParser(description="Drive SONIC keyboard interface")
    ap.add_argument("--motion-data", default=MOTION_DATA)
    ap.add_argument("--deploy-dir", default=str(DEPLOY_DIR))
    ap.add_argument("--play", nargs="*", help="play these motions then exit")
    ap.add_argument("--llm", action="store_true", help="route typed text through Nemotron")
    ap.add_argument("--echo", action="store_true", help="mirror deploy output")
    args = ap.parse_args()

    router = client = None
    suffix = ""
    if args.llm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import nemotron_router as router  # noqa: PLC0415
        client = router.build_client()
        # reference folders are the IL-ordered copies, named <motion>_il
        suffix = "_il"

    d = DeployDriver(args.deploy_dir, args.motion_data, echo=args.echo)
    try:
        d.start()
        d.start_policy()
        input("\npress 9 in the MuJoCo window, wait ~5s, then press ENTER here: ")

        found = d.list_motions()
        print(f"motions available: {', '.join(found)}")

        if args.play:
            for name in args.play:
                print(f"\n> {name}")
                if d.select(name):
                    d.play()
            return 0

        print("\nType a command, or 'quit'.\n")
        while True:
            try:
                cmd = input("command> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not cmd:
                continue
            if cmd.lower() in {"quit", "exit", "q"}:
                break

            if args.llm:
                res = router.route(client, cmd)
                motion = res.get("motion")
                if not motion:
                    print("  no matching motion")
                    continue
                print(f"  -> {motion} (confidence: {res.get('confidence')})")
                target = motion + suffix
            else:
                target = cmd

            if target not in found:
                print(f"  '{target}' not loaded. available: {', '.join(found)}")
                continue
            if d.select(target):
                d.play()
    finally:
        d.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
