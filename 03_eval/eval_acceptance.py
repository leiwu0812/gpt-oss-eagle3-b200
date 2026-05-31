"""Measure mean accepted-tokens-per-step (tau) on MT-Bench style prompts.

Uses vLLM EAGLE3 speculative decoding. Tau = 1 + num_accepted / num_drafts
(vLLM convention, includes bonus token). Requires disable_log_stats=False.
"""
import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def collect_spec_metrics(metrics, draft_len: int) -> dict:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * draft_len

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, val in enumerate(metric.values):
                if pos < len(acceptance_counts):
                    acceptance_counts[pos] += val

    tau = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    accept_rate = num_accepted_tokens / max(num_draft_tokens, 1)
    per_pos = [
        (acceptance_counts[i] / num_drafts if num_drafts > 0 else 0.0)
        for i in range(draft_len)
    ]
    return {
        "tau": tau,
        "acceptance_length": tau,
        "accept_rate": accept_rate,
        "num_drafts": int(num_drafts),
        "num_draft_tokens": int(num_draft_tokens),
        "num_accepted_tokens": int(num_accepted_tokens),
        "acceptance_per_pos": per_pos,
    }


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
        disable_log_stats=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    t0 = time.time()
    llm.generate(prompts, sp)
    dt = time.time() - t0
    stats = collect_spec_metrics(llm.get_metrics(), draft_len)
    stats.update({"wall_s": dt, "n": len(prompts)})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="~/models/gpt-oss-120b")
    ap.add_argument("--draft", required=True, help="EAGLE3 draft checkpoint dir")
    ap.add_argument("--label", default="draft", help="label in JSON output")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--draft_len", type=int, default=3)
    args = ap.parse_args()

    mtb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    prompts = [t for ex in mtb for t in ex["prompt"][:1]][: args.n]

    res = run(
        str(Path(args.target).expanduser()),
        str(Path(args.draft).expanduser()),
        prompts,
        args.draft_len,
    )
    res["label"] = args.label
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
