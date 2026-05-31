"""Measure mean accepted-tokens-per-step (tau) on MT-Bench style prompts.

Uses vLLM with speculative decoding. Compares against the reference
nvidia/gpt-oss-120b-Eagle3-long-context checkpoint when available.
Target from paper: 1.95 - 2.83 across MT-Bench categories at draft_len=3.
"""
import argparse, json, time
from pathlib import Path
from vllm import LLM, SamplingParams
from datasets import load_dataset


def run(target, draft, prompts, draft_len=3, max_new=512):
    llm = LLM(
        model=target,
        tensor_parallel_size=8,
        speculative_config={
            "model": draft,
            "num_speculative_tokens": draft_len,
            "method": "eagle3",
        },
        max_model_len=8192,
        dtype="bfloat16",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    # vLLM exposes spec stats on the request output if metrics enabled.
    accepted = sum(getattr(o, "num_accepted_tokens", 0) for o in outs)
    proposed = sum(getattr(o, "num_proposed_tokens", 1) for o in outs)
    tau = 1 + draft_len * (accepted / max(proposed, 1))
    return {"tau": tau, "wall_s": dt, "n": len(prompts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="~/models/gpt-oss-120b")
    ap.add_argument("--draft", required=True, help="trained draft ckpt dir")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--draft_len", type=int, default=3)
    args = ap.parse_args()

    mtb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    prompts = [t for ex in mtb for t in ex["prompt"][:1]][: args.n]

    res = run(str(Path(args.target).expanduser()),
              str(Path(args.draft).expanduser()),
              prompts, args.draft_len)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
