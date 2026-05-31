"""Sub-sample training prompts for EAGLE3 dry-run.

Default: Nemotron Post-Training V2 (500k recipe, multilingual x0.1).
Fallback: UltraChat-200k when Nemotron is gated/unavailable.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def load_nemotron(n: int, seed: int):
    random.seed(seed)
    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split="train", streaming=True)
    kept = []
    for ex in ds:
        cat = (ex.get("category") or "").lower()
        if "multilingual" in cat and random.random() >= 0.1:
            continue
        kept.append(ex)
        if len(kept) >= n * 3:
            break
    random.shuffle(kept)
    kept = kept[:n]
    rows = []
    for ex in kept:
        prompt = ex.get("input") or ex.get("prompt") or ex.get("messages")
        rows.append({"id": ex.get("id"), "prompt": prompt, "category": ex.get("category")})
    return rows


def load_ultrachat(n: int, seed: int):
    random.seed(seed)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    rows = []
    for ex in ds:
        msgs = ex.get("messages") or []
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        if not user_msgs:
            continue
        rows.append({
            "id": ex.get("prompt_id") or len(rows),
            "prompt": user_msgs[0]["content"],
            "category": "ultrachat",
        })
        if len(rows) >= n:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5_000, help="target sample count")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("~/eagle3-gptoss/01_data/prompts.jsonl").expanduser(),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--dataset",
        choices=["nemotron", "ultrachat", "auto"],
        default="auto",
        help="nemotron (paper recipe), ultrachat (open fallback), auto (try nemotron first)",
    )
    args = ap.parse_args()

    rows = None
    if args.dataset in ("nemotron", "auto"):
        try:
            rows = load_nemotron(args.n, args.seed)
            print(f"loaded {len(rows)} prompts from Nemotron V2")
        except Exception as e:
            print(f"Nemotron unavailable ({e}); falling back to UltraChat")
            if args.dataset == "nemotron":
                raise

    if rows is None:
        rows = load_ultrachat(args.n, args.seed)
        print(f"loaded {len(rows)} prompts from UltraChat")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
