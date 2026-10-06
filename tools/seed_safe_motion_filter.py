"""
seed_safe_motion_filter.py
==========================
Shortlist "safe for SONIC playback" motions from BONES-SEED metadata.

Loads metadata straight from the HF Hub -- does NOT download the
142k-motion dataset. You only download the handful of CSVs you pick.

Requires HF access to bones-studio/seed (gated: accept terms once on the
dataset page, then `huggingface-cli login` or export HF_TOKEN).

    pip install pandas pyarrow huggingface_hub
    python seed_safe_motion_filter.py

Outputs:
    seed_safe_shortlist.csv   -- full filtered set
    prints per-category top candidates
"""

import os
import sys
import pandas as pd

BASE = "hf://datasets/bones-studio/seed/metadata"
FPS = 120  # metadata durations are in 120fps frames

# duration window: long enough to read on camera, short enough for a demo clip
MIN_SEC, MAX_SEC = 2.5, 10.0


def load_metadata():
    """Try v003 first (per dataset file listing), fall back to v002."""
    last_err = None
    for name in ("seed_metadata_v004.parquet", "seed_metadata_v003.parquet", "seed_metadata_v002.parquet"):
        try:
            df = pd.read_parquet(f"{BASE}/{name}")
            print(f"Loaded {name}: {len(df):,} rows x {len(df.columns)} cols")
            return df
        except Exception as e:  # noqa: BLE001
            print(f"  {name} failed: {type(e).__name__}")
            last_err = e
    print("\nCould not load metadata. If this is a 401/403, the dataset is")
    print("gated -- accept the terms at huggingface.co/datasets/bones-studio/seed")
    print("then `huggingface-cli login` or export HF_TOKEN.")
    raise SystemExit(1) from last_err


def norm_flag(s):
    """content_* flags are inconsistently int/str/float across dumps."""
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def norm_props(s):
    """content_props is '0' when no prop, else an object descriptor."""
    return s.fillna("0").astype(str).str.strip().str.lower().isin({"0", "", "none", "nan"})


def main():
    df = load_metadata()

    # ---- show what we actually got, so filters can be adjusted if schema drifts
    for col in ("content_body_position", "content_type_of_movement", "category"):
        if col in df.columns:
            print(f"\n{col} -- top values:")
            print(df[col].value_counts().head(12).to_string())
        else:
            print(f"\nWARNING: column '{col}' not present")

    # ---- build the safety mask, one clause at a time so we can see the cost
    mask = pd.Series(True, index=df.index)
    steps = []

    def apply(name, cond):
        nonlocal mask
        before = int(mask.sum())
        mask = mask & cond
        steps.append((name, before, int(mask.sum())))

    if "is_mirror" in df:
        apply("not mirrored", ~df["is_mirror"].astype(bool))
    if "is_neutral" in df:
        apply("neutral style", pd.to_numeric(df["is_neutral"], errors="coerce").fillna(0) > 0)
    if "content_body_position" in df:
        apply("standing", df["content_body_position"].astype(str).str.lower().eq("standing"))
    if "content_horizontal_move" in df:
        apply("no horizontal travel", norm_flag(df["content_horizontal_move"]).eq(0))
    if "content_props" in df:
        apply("no props", norm_props(df["content_props"]))
    if "content_complex_action" in df:
        apply("not multi-phase", norm_flag(df["content_complex_action"]).eq(0))
    if "move_duration_frames" in df:
        secs = pd.to_numeric(df["move_duration_frames"], errors="coerce") / FPS
        apply(f"{MIN_SEC}-{MAX_SEC}s", secs.between(MIN_SEC, MAX_SEC))
    if "move_g1_mujoco_path" in df:
        apply("has G1 CSV", df["move_g1_mujoco_path"].notna())

    print("\nFilter funnel:")
    for name, before, after in steps:
        print(f"  {name:<24} {before:>7,} -> {after:>7,}")

    safe = df[mask].copy()
    if "move_duration_frames" in safe:
        safe["duration_sec"] = (
            pd.to_numeric(safe["move_duration_frames"], errors="coerce") / FPS
        ).round(2)

    print(f"\nSafe candidates: {len(safe):,}")

    # ---- vertical move is a soft signal, not a disqualifier:
    # squats/bows are vertical_move=1 and are exactly what we want.
    if "content_vertical_move" in safe:
        v = norm_flag(safe["content_vertical_move"])
        print(f"  of which vertical motion (squat/bow-like): {int(v.sum()):,}")

    cols = [c for c in [
        "filename", "content_short_description", "content_type_of_movement",
        "category", "package", "duration_sec", "content_vertical_move",
        "move_g1_mujoco_path",
    ] if c in safe.columns]

    out = "seed_safe_shortlist.csv"
    safe[cols].to_csv(out, index=False)
    print(f"\nWrote {out}")

    # ---- browse by category so you can pick 3-4 visually distinct motions
    if "category" in safe.columns:
        print("\nSafe motions by category:")
        print(safe["category"].value_counts().head(15).to_string())

        interesting = [
            "Gestures", "Looking and Pointing", "Communication",
            "Baseline", "Basic Locomotion Neutral",
        ]
        for cat in interesting:
            sub = safe[safe["category"] == cat]
            if sub.empty:
                continue
            print(f"\n--- {cat} ({len(sub):,} safe) ---")
            show = [c for c in ("filename", "content_short_description", "duration_sec") if c in sub.columns]
            print(sub[show].head(15).to_string(index=False))

    # ---- keyword probe for the specific behaviours we want in the demo
    if "content_short_description" in safe.columns:
        desc = safe["content_short_description"].fillna("").astype(str).str.lower()
        print("\nKeyword hits (safe set only):")
        for kw in ["wave", "point", "squat", "crouch", "bow", "nod",
                   "turn", "salute", "clap", "shrug", "stretch", "reach"]:
            hits = safe[desc.str.contains(kw, na=False)]
            if hits.empty:
                print(f"  {kw:<9} 0")
                continue
            sample = hits["filename"].head(3).tolist()
            print(f"  {kw:<9} {len(hits):>5}   e.g. {', '.join(sample)}")


if __name__ == "__main__":
    main()
