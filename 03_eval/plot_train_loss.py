#!/usr/bin/env python3
"""Plot EAGLE3 draft training loss/acc from trainer_state.json and logs."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/workspace/eagle3-gptoss")
if not ROOT.exists():
    ROOT = Path.home() / "eagle3-gptoss"
TS_PATH = ROOT / "ckpt/dry-run/trainer_state.json"
LOG_PATH = Path("/tmp/eagle3_train_step4.log")
if not LOG_PATH.exists():
    LOG_PATH = ROOT / "logs" / "train_step4.log"
OUT_DIR = ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = json.loads(TS_PATH.read_text())
steps, losses, accs = [], [], []

for entry in ts["log_history"]:
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])
    if "train_loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["train_loss"])

# Parse per-step acc from log if present
acc_by_step = {}
if LOG_PATH.exists():
    text = LOG_PATH.read_text(errors="replace")
    for m in re.finditer(
        r"Step (\d+) Training Acc: \[([\d., ]+)\]", text
    ):
        step = int(m.group(1))
        accs_pos = [float(x.strip()) for x in m.group(2).split(",")]
        acc_by_step[step] = sum(accs_pos) / len(accs_pos)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(steps, losses, "o-", color="#2563eb", linewidth=2, markersize=8)
for x, y in zip(steps, losses):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("Global step")
ax.set_ylabel("Loss")
ax.set_title("EAGLE3 draft training loss (dry-run, 8 steps)")
ax.set_xticks(range(1, max(steps) + 1))
ax.grid(True, alpha=0.3)

ax2 = axes[1]
if acc_by_step:
    acc_steps = sorted(acc_by_step)
    acc_vals = [acc_by_step[s] for s in acc_steps]
    ax2.plot(acc_steps, acc_vals, "s-", color="#16a34a", linewidth=2, markersize=8)
    for x, y in zip(acc_steps, acc_vals):
        ax2.annotate(f"{y*100:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax2.set_ylabel("Mean train acc (3 draft positions)")
else:
    ax2.text(0.5, 0.5, "No per-step acc in log", ha="center", va="center", transform=ax2.transAxes)
ax2.set_xlabel("Global step")
ax2.set_title("Training accuracy")
ax2.set_xticks(range(1, max(steps) + 1))
ax2.grid(True, alpha=0.3)

fig.tight_layout()
png = OUT_DIR / "dry_run_loss_curve.png"
csv = OUT_DIR / "dry_run_loss_curve.csv"
fig.savefig(png, dpi=150)

with csv.open("w") as f:
    f.write("step,loss\n")
    for s, l in zip(steps, losses):
        f.write(f"{s},{l}\n")
    if acc_by_step:
        f.write("\nstep,mean_train_acc\n")
        for s in sorted(acc_by_step):
            f.write(f"{s},{acc_by_step[s]}\n")

print(json.dumps({"png": str(png), "csv": str(csv), "steps": steps, "losses": losses, "acc_by_step": acc_by_step}))
