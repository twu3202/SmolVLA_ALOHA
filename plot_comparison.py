#!/usr/bin/env python
"""
Cross-dataset comparison plot.

Reads all eval_*.npz files in eval_output/ and produces a single
summary figure comparing SmolVLA performance across datasets/robots.

Panels:
  1. Overall L2 error bar chart (one bar per dataset)
  2. Mean MAE per dataset
  3. Normalised per-dim MAE heatmap (dim / mean, so datasets are comparable)
  4. Loss curve comparison (reads loss_log from checkpoints)

Usage:
    /opt/anaconda3/envs/lerobot/bin/python plot_comparison.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT_DIR  = Path("./eval_output")
CKPT_DIR = Path("./checkpoints")
OUT_FILE = OUT_DIR / "comparison_all_datasets.png"

# Pretty labels and colors per dataset
DATASET_LABELS = {
    # Sim human
    "aloha_transfer":              "ALOHA Transfer\n(sim-human, 50ep)",
    "aloha_insertion":             "ALOHA Insertion\n(sim-human, 50ep)",
    "aloha_multitask":             "ALOHA Multi-task\n(sim-human, 100ep)",
    # Sim scripted
    "aloha_transfer_scripted":     "ALOHA Transfer\n(sim-script, 50ep)",
    "aloha_insertion_scripted":    "ALOHA Insertion\n(sim-script, 50ep)",
    "aloha_multitask_scripted":    "ALOHA Multi-task\n(sim-script, 100ep)",
    # Real robot original
    "aloha_static_coffee":         "ALOHA Coffee\n★REAL★",
    "aloha_static_battery":        "ALOHA Battery\n★REAL★",
    # Real robot extended
    "aloha_static_cups_open":      "ALOHA Cups\n★REAL★",
    "aloha_static_towel":          "ALOHA Towel\n★REAL deform★",
    "aloha_static_ziploc_slide":   "ALOHA Ziploc\n★REAL fine★",
    # xArm
    "xarm_lift":                   "xArm Lift\n(medium, 800ep)",
    "xarm_push":                   "xArm Push\n(medium, 800ep)",
    "xarm_lift_replay":            "xArm Lift\n(replay, 800ep)",
    "xarm_push_replay":            "xArm Push\n(replay, 800ep)",
    # PushT
    "pusht":                       "PushT\n(2-DOF, 206ep)",
}
# Group colors: ALOHA sim human=blue, scripted=cyan, real=red/orange, xArm=green, pusht=purple
GROUP_COLORS = {
    "aloha_transfer":              "#1f77b4",
    "aloha_insertion":             "#4499cc",
    "aloha_multitask":             "#2255aa",
    "aloha_transfer_scripted":     "#17becf",
    "aloha_insertion_scripted":    "#39d6e8",
    "aloha_multitask_scripted":    "#0099aa",
    "aloha_static_coffee":         "#d62728",
    "aloha_static_battery":        "#e05555",
    "aloha_static_cups_open":      "#ff7f0e",
    "aloha_static_towel":          "#e8963a",
    "aloha_static_ziploc_slide":   "#cc6622",
    "xarm_lift":                   "#2ca02c",
    "xarm_push":                   "#5cb85c",
    "xarm_lift_replay":            "#1a7a1a",
    "xarm_push_replay":            "#3aaa3a",
    "pusht":                       "#9467bd",
}
COLORS = list(GROUP_COLORS.values())


def load_eval_results():
    results = {}
    for npz_path in sorted(OUT_DIR.glob("eval_*.npz")):
        name = npz_path.stem   # e.g. eval_aloha_insertion_step3000
        parts = name.split("_step")
        dataset_key = parts[0].replace("eval_", "")
        step = int(parts[1]) if len(parts) > 1 else 0
        data = np.load(npz_path, allow_pickle=True)
        key = str(data.get("dataset", dataset_key))
        if not key or key == "None":
            key = dataset_key
        # Keep latest step per dataset
        if key not in results or step > results[key]["step"]:
            results[key] = {
                "per_dim_mae": data["per_dim_mae"],
                "overall_l2":  float(data["overall_l2"]),
                "step":        step,
                "label":       DATASET_LABELS.get(key, key),
            }
    return results


def load_loss_curves():
    curves = {}
    for ckpt_dir in sorted(CKPT_DIR.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        key = ckpt_dir.name
        # Find latest checkpoint with loss_log
        for pt_path in sorted(ckpt_dir.glob("step_*.pt"), reverse=True):
            try:
                import torch
                ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
                if "loss_log" in ckpt and ckpt["loss_log"]:
                    curves[key] = {
                        "log":   ckpt["loss_log"],
                        "label": DATASET_LABELS.get(key, key),
                    }
                    break
            except Exception:
                continue
    return curves


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results    = load_eval_results()
    loss_curves = load_loss_curves()

    if not results and not loss_curves:
        print("No eval results or checkpoints found. Run training + eval first.")
        return

    print(f"Found eval results for: {list(results.keys())}")
    print(f"Found loss curves for:  {list(loss_curves.keys())}")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 13))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)
    ax_l2    = fig.add_subplot(gs[0, 0])
    ax_mae   = fig.add_subplot(gs[0, 1])
    ax_heat  = fig.add_subplot(gs[0, 2])
    ax_loss  = fig.add_subplot(gs[1, :])

    fig.suptitle("SmolVLA Cross-Dataset Reproduction — Performance Comparison",
                 fontsize=14, fontweight="bold", y=0.99)

    # Sort by group then name
    group_order = ["aloha_transfer","aloha_insertion","aloha_multitask",
                   "aloha_transfer_scripted","aloha_insertion_scripted","aloha_multitask_scripted",
                   "aloha_static_coffee","aloha_static_battery",
                   "aloha_static_cups_open","aloha_static_towel","aloha_static_ziploc_slide",
                   "xarm_lift","xarm_push","xarm_lift_replay","xarm_push_replay",
                   "pusht"]
    keys   = [k for k in group_order if k in results] + \
             [k for k in sorted(results.keys()) if k not in group_order]
    labels = [results[k]["label"] for k in keys]
    n      = len(keys)
    clrs   = [GROUP_COLORS.get(k, "#888888") for k in keys]

    # ── Panel 1: Overall L2 ───────────────────────────────────────────────────
    if results:
        l2_vals = [results[k]["overall_l2"] for k in keys]
        bars = ax_l2.bar(range(n), l2_vals, color=clrs, edgecolor="white", width=0.6)
        ax_l2.set_xticks(range(n))
        ax_l2.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax_l2.set_ylabel("L2 error (original units / step)")
        ax_l2.set_title("Overall L2 Error per Timestep", fontsize=10, fontweight="bold")
        ax_l2.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, l2_vals):
            ax_l2.text(b.get_x() + b.get_width()/2, v + max(l2_vals)*0.02,
                       f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # ── Panel 2: Mean MAE ─────────────────────────────────────────────────────
    if results:
        mae_vals = [results[k]["per_dim_mae"].mean() for k in keys]
        bars2 = ax_mae.bar(range(n), mae_vals, color=clrs, edgecolor="white", width=0.6)
        ax_mae.set_xticks(range(n))
        ax_mae.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax_mae.set_ylabel("Mean MAE across all joints")
        ax_mae.set_title("Mean Per-Joint MAE", fontsize=10, fontweight="bold")
        ax_mae.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars2, mae_vals):
            ax_mae.text(b.get_x() + b.get_width()/2, v + max(mae_vals)*0.02,
                        f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # ── Panel 3: Normalised per-dim heatmap ───────────────────────────────────
    if results:
        # Normalise each dataset's MAE vector by its mean so shapes are comparable
        max_dim = max(len(results[k]["per_dim_mae"]) for k in keys)
        heat = np.full((n, max_dim), np.nan)
        for i, k in enumerate(keys):
            v = results[k]["per_dim_mae"]
            norm_v = v / (v.mean() + 1e-9)
            heat[i, :len(v)] = norm_v

        im = ax_heat.imshow(heat, aspect="auto", cmap="RdYlGn_r",
                            vmin=0.0, vmax=3.0)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(labels, fontsize=8)
        ax_heat.set_xlabel("Joint / action dimension index")
        ax_heat.set_title("Normalised per-dim MAE\n(1.0 = dataset mean)",
                          fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # ── Panel 4: Loss curves ─────────────────────────────────────────────────
    if loss_curves:
        smooth = 50   # rolling average window
        curve_order = [k for k in group_order if k in loss_curves] + \
                      [k for k in sorted(loss_curves.keys()) if k not in group_order]
        for key in curve_order:
            info = loss_curves[key]
            log = np.array(info["log"])
            if len(log) < smooth:
                smoothed = log
                x = np.arange(len(log))
            else:
                smoothed = np.convolve(log, np.ones(smooth)/smooth, mode="valid")
                x = np.arange(smooth-1, len(log))
            color = GROUP_COLORS.get(key, "#888888")
            ax_loss.plot(x, smoothed, lw=1.8, color=color,
                         label=f"{DATASET_LABELS.get(key, key).replace(chr(10),' ')} "
                               f"(final={log[-100:].mean():.3f})")

        ax_loss.set_xlabel("Training step")
        ax_loss.set_ylabel("Flow-matching loss")
        ax_loss.set_title("Training Loss Curves (smoothed, window=50)",
                          fontsize=10, fontweight="bold")
        ax_loss.legend(fontsize=9, loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim(bottom=0)

    plt.savefig(OUT_FILE, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {OUT_FILE}")

    # Print summary table
    print(f"\n{'Dataset':<22} {'Step':>6} {'L2':>8} {'Mean MAE':>10}")
    print("─" * 52)
    for k in keys:
        r = results[k]
        print(f"  {k:<20} {r['step']:>6} {r['overall_l2']:>8.4f} "
              f"{r['per_dim_mae'].mean():>10.4f}")


if __name__ == "__main__":
    main()
