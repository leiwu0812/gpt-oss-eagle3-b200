"""Regenerate responses with gpt-oss-120b using vLLM.

Each sample uses randomized reasoning_effort in {low, medium, high}
and temperature ~ U(0, 1) per paper Appendix H.
"""
import argparse, json, random
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/gpt-oss-120b")
    ap.add_argument("--prompts", type=Path, default=Path("~/eagle3-gptoss/01_data/prompts.jsonl").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/eagle3-gptoss/01_data/regen.jsonl").expanduser())
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_new", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    model_path = str(Path(args.model).expanduser())
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )

    items = [json.loads(l) for l in args.prompts.open()]
    random.seed(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as fout:
        for batch in chunked(items, args.batch):
            prompts, sps, metas = [], [], []
            for ex in batch:
                effort = random.choice(["low", "medium", "high"])
                temp = random.random()
                user = ex["prompt"] if isinstance(ex["prompt"], str) else json.dumps(ex["prompt"])
                msgs = [
                    {"role": "system", "content": f"reasoning: {effort}"},
                    {"role": "user", "content": user},
                ]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                sps.append(SamplingParams(temperature=temp, max_tokens=args.max_new, top_p=1.0))
                metas.append({"id": ex.get("id"), "effort": effort, "temperature": temp})

            outs = llm.generate(prompts, sps)
            for meta, p, o in zip(metas, prompts, outs):
                fout.write(json.dumps({
                    **meta,
                    "prompt_text": p,
                    "response": o.outputs[0].text,
                }, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"wrote regenerated jsonl -> {args.out}")


if __name__ == "__main__":
    main()
