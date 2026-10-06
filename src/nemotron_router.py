"""
nemotron_router.py
==================
Turns a natural-language command into a motion choice using an NVIDIA
Nemotron model served on Nebius Token Factory.

  "do the moonwalk"  ->  {"motion": "moonwalk", "confidence": "high"}

Token Factory is OpenAI-compatible:
  base_url = https://api.tokenfactory.nebius.com/v1/
  api key  = NEBIUS_API_KEY env var  (get one at tokenfactory.nebius.com)

  pip install openai


USAGE
  export NEBIUS_API_KEY=...
  python nemotron_router.py "wave at me"
  python nemotron_router.py --list-models
  python nemotron_router.py --selftest
"""

import argparse
import json
import os
import re
import sys

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Confirm the exact ID in the Token Factory console -- names change.
# Nemotron families seen on the platform: Nemotron-3-Nano / Super / Ultra.
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


MOTION_CATALOG = {
    "walk":     "walk forward a few steps",
    "moonwalk": "moonwalk, slide backwards",
    "wave": "wave hello — greeting someone, saying hi, waving at a person",
    "stretch":  "stretch — reach both arms up overhead and stretch the body",
}


SYSTEM_PROMPT = """You route robot commands to a fixed motion library.

Available motions:
{catalog}

Reply with ONLY a JSON object, no prose, no markdown fences:
{{"motion": "<one of: {names}, or null>", "confidence": "high|low"}}

Use null when no motion is a reasonable match. Do not invent motion names."""


def build_client():
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY not set")
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=key)


def extract_text(message):
    """Reasoning models may put the answer in reasoning_content."""
    text = (getattr(message, "content", None) or "").strip()
    if text:
        return text
    return (getattr(message, "reasoning_content", None) or "").strip()


def parse_json(text):
    """Pull the last JSON object out of a possibly chatty response."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def route(client, command, model=DEFAULT_MODEL, verbose=False):
    names = ", ".join(MOTION_CATALOG)
    catalog = "\n".join(f"- {k}: {v}" for k, v in MOTION_CATALOG.items())
    system = SYSTEM_PROMPT.format(catalog=catalog, names=names)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": command},
        ],
        temperature=0.1,
        max_tokens=2000,      # headroom for the reasoning trace
    )

    msg = resp.choices[0].message
    text = extract_text(msg)
    if verbose:
        print(f"--- raw ---\n{text}\n-----------")

    parsed = parse_json(text)
    if not parsed:
        return {"motion": None, "confidence": "low", "error": "unparseable response"}

    motion = parsed.get("motion")
    if motion is not None and motion not in MOTION_CATALOG:
        return {"motion": None, "confidence": "low",
                "error": f"model returned unknown motion '{motion}'"}
    return {"motion": motion, "confidence": parsed.get("confidence", "low")}


def selftest(client, model):

    cases = [
        ("do the moonwalk", "moonwalk"),
        ("slide backwards like michael jackson", "moonwalk"),
        ("walk forward", "walk"),
        ("say hi to me", "wave"),
        ("make me a sandwich", None),
    ]
    passed = 0
    for cmd, expected in cases:
        got = route(client, cmd, model)
        ok = got.get("motion") == expected
        passed += ok
        print(f"  [{'pass' if ok else 'FAIL'}] {cmd!r} -> {got.get('motion')} "
              f"(expected {expected})")
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    client = build_client()

    if args.list_models:
        for m in client.models.list().data:
            if "emotron" in m.id:
                print(f"  {m.id}")
        return 0

    if args.selftest:
        return selftest(client, args.model)

    if not args.command:
        ap.error("give a command, or use --selftest / --list-models")

    result = route(client, args.command, args.model, args.verbose)
    print(json.dumps(result))
    return 0 if result.get("motion") else 1


if __name__ == "__main__":
    sys.exit(main())
