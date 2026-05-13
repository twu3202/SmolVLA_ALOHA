#!/usr/bin/env python
"""
Side-by-side comparison of RT-1 vs SmolVLA on the same datasets.

Reads:
  ../eval_output/eval_{DATASET}_step*.npz          (SmolVLA results)
  ./eval_output/eval_rt1_{DATASET}_step*.npz       (RT-1 results)

Produces:
  ./eval_output/rt1_vs_smolvla.png      (bar chart head-to-head + loss curves)
"""

import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOL_DIR = ROOT / "eval_output"
RT1_DIR  = ROOT / "RT1_repro" / "eval_output"
RT1_CKPT = ROOT / "RT1_repro" / "checkpoints"
OUT      = RT1_DIR / "rt1_vs_smolvla.png"
RT1_DIR.mkdir(parents=True, exist_ok=True)


def latest_npz(d: Path, prefix: str) -> dict:
    """Return latest-step result for each dataset under directory `d`."""
    results = {}
    for p in sorted(d.glob(f"{prefix}*.npz")):
        name = p.stem.replace(prefix, "")
        if "_step" not in name: continue
        dskey, step = name.rsplit("_step", 1)
        try: step = int(step)
        except: continue
        data = np.load(p, allow_pickle=True)
        if dskey not in results or step > results[dskey]["step"]:
            results[dskey] = {
                "l2":   float(data["overall_l2"]),
                "mae":  np.asarray(data["per_dim_mae"]).mean(),
                "step": step,
            }
    return results


def load_rt1_loss(dataset: str):
    ckpt_dir = RT1_CKPT / dataset
    if not ckpt_dir.exists(): return None
    pts = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    if not pts: return None
    import torch
    try:
        ck = torch.load(pts[0], map_location="cpu", weights_only=False)
        return ck.get("loss_log", None)
    except Exception:
        return None


def main():
    smol = latest_npz(SMOL_DIR, "eval_")
    rt1  = latest_npz(RT1_DIR,  "eval_rt1_")

    # Datasets present in both
    keys = sorted(set(rt1.keys()) & set(smol.keys()))
    if not keys:
        print("No overlap between RT-1 and SmolVLA results yet.")
        return

    print(f"Datasets compared: {keys}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1.2]})

    # ── Panel 1: L2 head-to-head ────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(keys))
    w = 0.38
    smol_l2 = [smol[k]["l2"] for k in keys]
    rt1_l2  = [rt1[k]["l2"]  for k in keys]
    b1 = ax.bar(x - w/2, smol_l2, w, label="SmolVLA (450M)", color="#1f77b4", edgecolor="white")
    b2 = ax.bar(x + w/2, rt1_l2,  w, label="RT-1 (243M)",    color="#d62728", edgecolor="white")
    for b, v in zip(b1, smol_l2):
        ax.text(b.get_x() + b.get_width()/2, v + max(smol_l2 + rt1_l2)*0.02,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    for b, v in zip(b2, rt1_l2):
        ax.text(b.get_x() + b.get_width()/2, v + max(smol_l2 + rt1_l2)*0.02,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overall L2 (original units)")
    ax.set_title("L2 head-to-head", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Mean MAE ──────────────────────────────────────────────────
    ax = axes[1]
    smol_mae = [smol[k]["mae"] for k in keys]
    rt1_mae  = [rt1[k]["mae"]  for k in keys]
    ax.bar(x - w/2, smol_mae, w, label="SmolVLA", color="#1f77b4", edgecolor="white")
    ax.bar(x + w/2, rt1_mae,  w, label="RT-1",    color="#d62728", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean per-dim MAE")
    ax.set_title("Mean MAE head-to-head", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: RT-1 loss curves ──────────────────────────────────────────
    ax = axes[2]
    for k in keys:
        log = load_rt1_loss(k)
        if log is None: continue
        log = np.asarray(log)
        if len(log) >= 50:
            smoothed = np.convolve(log, np.ones(50)/50, mode="valid")
            xs = np.arange(49, len(log))
            ax.plot(xs, smoothed, lw=1.5, label=f"{k} (f={log[-100:].mean():.2f})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("RT-1 training loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("RT-1 (lucidrains port) vs SmolVLA on shared LeRobot v3 datasets",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {OUT}")

    print(f"\n{'Dataset':<26} {'SmolVLA L2':>11} {'RT-1 L2':>10} {'Winner':>10}")
    print("─" * 60)
    for k in keys:
        winner = "SmolVLA" if smol[k]["l2"] < rt1[k]["l2"] else "RT-1"
        diff = (rt1[k]["l2"] - smol[k]["l2"]) / smol[k]["l2"] * 100
        print(f"  {k:<24} {smol[k]['l2']:>11.4f} {rt1[k]['l2']:>10.4f} {winner:>10}  ({diff:+.0f}%)")


if __name__ == "__main__":
    main()
