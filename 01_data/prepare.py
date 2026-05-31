"""Sub-sample Nemotron Post-Training Dataset V2.

Paper recipe: 500k samples, multilingual category weighted 10x lower.
For dry-run set N=5000.
"""
import argparse, json, random
from pathlib import Path
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5_000, help="target sample count")
    ap.add_argument("--out", type=Path, default=Path("~/eagle3-gptoss/01_data/prompts.jsonl").expanduser())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split="train", streaming=True)

    # Stream-filter with category-aware reservoir sampling.
    # multilingual is downweighted by keeping it at 0.1 probability.
    kept = []
    for ex in ds:
        cat = (ex.get("category") or "").lower()
        if "multilingual" in cat and random.random() >= 0.1:
            continue
        kept.append(ex)
        if len(kept) >= args.n * 3:  # collect a pool then shuffle/truncate
            break

    random.shuffle(kept)
    kept = kept[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in kept:
            # Normalize to {id, prompt, category}
            prompt = ex.get("input") or ex.get("prompt") or ex.get("messages")
            f.write(json.dumps({
                "id": ex.get("id"),
                "prompt": prompt,
                "category": ex.get("category"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
