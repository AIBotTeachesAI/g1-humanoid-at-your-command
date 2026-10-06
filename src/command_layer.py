"""
command_layer.py
================
The glue: natural-language command -> Nemotron -> SONIC.

    "wave at me"
       -> nemotron_router   (Nemotron on Nebius Token Factory)
       -> {"motion": "wave"}
       -> sonic_motion_publisher  (ZMQ pose topic)
       -> SONIC -> Unitree G1 in MuJoCo

Frame indices stay monotonic across commands within a session, which is what
SONIC's streamed-motion mode expects -- a jump backwards reads as a
discontinuity.

USAGE
  # one command, no ZMQ -- proves routing + motion loading
  uv run python src/command_layer.py --dry-run "wave at me"

  # interactive session, no ZMQ
  uv run python src/command_layer.py --dry-run

  # live against a running SONIC deploy
  uv run python src/command_layer.py "wave at me"
  uv run python src/command_layer.py            # interactive

  # skip the LLM, name the motion directly
  uv run python src/command_layer.py --no-llm wave
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sonic_motion_publisher import (  # noqa: E402
    MOTIONS,
    SonicPublisher,
    load_motion,
    sanity_check,
    stream_motion,
)

import nemotron_router  # noqa: E402


class CommandLayer:
    """Holds the session: one publisher, one running frame counter."""

    def __init__(self, dry_run=False, host="*", port=5556, model=None,
                 use_llm=True, verbose=False):
        self.dry_run = dry_run
        self.use_llm = use_llm
        self.verbose = verbose
        self.model = model or nemotron_router.DEFAULT_MODEL
        self.frame_idx = 0
        self.started = False
        self.cache = {}

        self.client = None
        if use_llm and not self._catalog_matches():
            print("WARNING: MOTION_CATALOG and MOTIONS disagree -- "
                  "the router may pick a motion that won't load.")
            print(f"  router : {sorted(nemotron_router.MOTION_CATALOG)}")
            print(f"  publisher: {sorted(MOTIONS)}")
        if use_llm:
            self.client = nemotron_router.build_client()

        self.pub = None
        if not dry_run:
            self.pub = SonicPublisher(host=host, port=port, verbose=verbose)

    @staticmethod
    def _catalog_matches():
        return set(nemotron_router.MOTION_CATALOG) == set(MOTIONS)

    def resolve(self, command):
        """command text -> motion name, or None."""
        if not self.use_llm:
            name = command.strip()
            return name if name in MOTIONS else None

        result = nemotron_router.route(
            self.client, command, self.model, verbose=self.verbose
        )
        if result.get("error"):
            print(f"  router error: {result['error']}")
        motion = result.get("motion")
        if motion:
            print(f"  -> {motion}  (confidence: {result.get('confidence')})")
        return motion

    def get_motion(self, name):
        """Load + validate, cached across repeats within a session."""
        if name in self.cache:
            return self.cache[name]
        jp, jv, bq = load_motion(MOTIONS[name])
        if not sanity_check(name, jp, jv, bq):
            raise ValueError(f"{name} failed sanity check")
        self.cache[name] = (jp, jv, bq)
        return self.cache[name]

    def handle(self, command):
        print(f"\n> {command}")

        motion = self.resolve(command)
        if motion is None:
            print("  no matching motion -- nothing sent")
            return False
        if motion not in MOTIONS:
            print(f"  '{motion}' is not in MOTIONS -- nothing sent")
            return False

        jp, jv, bq = self.get_motion(motion)

        if self.dry_run:
            print(f"  [dry run] would stream {len(jp)} frames "
                  f"from index {self.frame_idx}")
            self.frame_idx += len(jp)
            return True

        # start command once per session; the robot must already be
        # stabilized in the sim (press 9 and let it settle ~5s)
        if not self.started:
            self.pub.send_command(start=True, stop=False, planner=False)
            self.started = True
            import time
            time.sleep(0.5)

        print(f"  streaming {len(jp)} frames from index {self.frame_idx}...")
        self.frame_idx = stream_motion(
            self.pub, jp, jv, bq, start_frame=self.frame_idx
        )
        print("  done")
        return True

    def close(self):
        if self.pub:
            self.pub.close()


def interactive(layer):
    print("\nMotions:", ", ".join(MOTIONS))
    print("Type a command, or 'quit'.\n")
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
        try:
            layer.handle(cmd)
        except Exception as e:  # noqa: BLE001
            print(f"  error: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Natural language -> SONIC motion")
    ap.add_argument("command", nargs="*", help="command text; omit for interactive")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and load, but send nothing")
    ap.add_argument("--no-llm", action="store_true",
                    help="treat the argument as a motion name, skip Nemotron")
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    layer = CommandLayer(
        dry_run=args.dry_run,
        host=args.host,
        port=args.port,
        model=args.model,
        use_llm=not args.no_llm,
        verbose=args.verbose,
    )

    try:
        if args.command:
            ok = layer.handle(" ".join(args.command))
            return 0 if ok else 1
        interactive(layer)
        return 0
    finally:
        layer.close()


if __name__ == "__main__":
    sys.exit(main())
