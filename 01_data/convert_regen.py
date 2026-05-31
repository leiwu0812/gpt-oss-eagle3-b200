"""Convert regenerate.py output to ModelOpt conversation JSONL.

ModelOpt collect_hidden_states_hf.py and launch_train.sh expect
conversation_id + conversations fields.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.inp.open() as fin, args.out.open("w") as fout:
        for line in fin:
            ex = json.loads(line)
            user = ex.get("prompt_text") or ex.get("prompt") or ""
            assistant = ex.get("response") or ""
            cid = ex.get("id", n)
            row = {
                "id": cid,
                "conversation_id": str(cid),
                "conversations": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"converted {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
