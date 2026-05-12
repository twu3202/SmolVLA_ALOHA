#!/usr/bin/env python
"""
Generic open-loop evaluation for any dataset trained with train_generic.py.

Metrics:
  - Per-joint / per-dimension MAE in original (unnormalised) units
  - Overall L2 error per timestep
  - Predicted vs GT trajectory plots

Usage:
    DATASET=aloha_insertion STEP=3000 \
        /opt/anaconda3/envs/lerobot/bin/python eval_generic.py

    DATASET=xarm_lift STEP=5000 \
        /opt/anaconda3/envs/lerobot/bin/python eval_generic.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

LEROBOT_SRC = "/Users/r/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from dataset_configs import DATASET_CONFIGS
from train_generic import (
    make_smolvla_config, load_parquet_frames, decode_frame,
    get_task_description, LeRobotDataset,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET   = os.environ.get("DATASET", "aloha_insertion")
STEP      = int(os.environ.get("STEP", "3000"))
N_EVAL_EP = int(os.environ.get("N_EVAL_EP", "10"))
DATA_DIR  = Path("./Data")
CKPT_BASE = Path("./checkpoints")
OUT_DIR   = Path("./eval_output")
# ──────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = {
    14: ["L_waist","L_shoulder","L_elbow","L_forearm","L_wrist_a","L_wrist_r","L_gripper",
         "R_waist","R_shoulder","R_elbow","R_forearm","R_wrist_a","R_wrist_r","R_gripper"],
    4:  ["x","y","z","gripper"],
    3:  ["x","y","z"],
    2:  ["x","y"],
}


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def build_obs_batch(img_np, state_raw, state_mean, state_std,
                    tokenizer, device, task_desc, cfg):
    import torch
    img = torch.from_numpy(img_np).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)
    state = (state_raw - state_mean) / (state_std + 1e-8)
    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    toks = tokenizer(task_desc + "\n", return_tensors="pt",
                     padding="max_length", max_length=48, truncation=True)
    return {
        cfg["image_key"]:            img,
        cfg["state_key"]:            state,
        OBS_LANGUAGE_TOKENS:         toks["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: toks["attention_mask"].bool().to(device),
    }


def eval_episode(model, tokenizer, df_ep, dataset_dir, task_desc,
                 state_mean, state_std, act_mean, act_std, device, cfg):
    model.reset()
    preds, gts = [], []
    for t in range(len(df_ep)):
        row        = df_ep.iloc[t]
        ep_idx     = int(row["episode_index"])
        global_idx = int(row["index"])
        try:
            img_np = decode_frame(dataset_dir, global_idx,
                                  cfg["image_key"], ep_idx)
        except Exception:
            img_np = np.zeros((cfg["image_h"], cfg["image_w"], 3), dtype=np.uint8)

        state_raw = np.array(row[cfg["state_key"]], dtype=np.float32)
        batch = build_obs_batch(img_np, state_raw, state_mean, state_std,
                                tokenizer, device, task_desc, cfg)
        pred_norm = model.select_action(batch).squeeze(0).cpu().numpy()
        preds.append(pred_norm * act_std + act_mean)
        gts.append(np.array(row[cfg["action_key"]], dtype=np.float32) * act_std + act_mean)

    return np.stack(preds), np.stack(gts)


def main():
    cfg    = DATASET_CONFIGS[DATASET]
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_BASE / DATASET / f"step_{STEP:06d}.pt"
    print(f"=== Eval: {DATASET}  step={STEP} ===")
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    # ── Load stats ─────────────────────────────────────────────────────────────
    with open(CKPT_BASE / DATASET / "dataset_stats.json") as f:
        stats = json.load(f)
    state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
    state_std  = np.array(stats["observation.state"]["std"],  dtype=np.float32)
    act_mean   = np.array(stats["action"]["mean"], dtype=np.float32)
    act_std    = np.array(stats["action"]["std"],  dtype=np.float32)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_cfg = make_smolvla_config(cfg, device)
    model_cfg.load_vlm_weights = False
    model = SmolVLAPolicy(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["policy_state"])
    model.eval()
    print(f"Loaded step={ckpt['step']}  train_loss={ckpt['loss']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    local_names = cfg["local_name"] if isinstance(cfg["local_name"], list) \
                  else [cfg["local_name"]]
    all_preds, all_gts = [], []

    for name in local_names:
        ddir      = DATA_DIR / name
        task_desc = get_task_description(ddir)
        df        = load_parquet_frames(ddir)
        episodes  = sorted(df["episode_index"].unique())
        eval_eps  = episodes[-N_EVAL_EP:]
        print(f"\nDataset: {name}  task: {task_desc[:60]}")
        print(f"Evaluating {len(eval_eps)} episodes ...")

        for ep_idx in eval_eps:
            df_ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            p, g  = eval_episode(model, tokenizer, df_ep, ddir, task_desc,
                                 state_mean, state_std, act_mean, act_std,
                                 device, cfg)
            all_preds.append(p); all_gts.append(g)
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

    preds = np.concatenate(all_preds, axis=0)   # (N, action_dim)
    gts   = np.concatenate(all_gts,   axis=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    action_dim  = cfg["action_dim"]
    per_dim_mae = np.abs(preds - gts).mean(axis=0)
    overall_l2  = np.sqrt(((preds - gts) ** 2).sum(axis=1)).mean()
    joint_names = JOINT_NAMES.get(action_dim, [f"dim{i}" for i in range(action_dim)])

    print(f"\n{'='*55}")
    print(f"Results  n={len(preds)} steps")
    print(f"{'='*55}")
    print(f"  Overall L2  : {overall_l2:.4f}")
    print(f"  Mean MAE    : {per_dim_mae.mean():.4f}")
    for name, mae in zip(joint_names, per_dim_mae):
        bar = "█" * int(mae * 60)
        print(f"  {name:<14} {mae:.4f}  {bar}")

    # ── Save numeric ──────────────────────────────────────────────────────────
    np.savez(OUT_DIR / f"eval_{DATASET}_step{STEP}.npz",
             preds=preds, gts=gts,
             per_dim_mae=per_dim_mae, overall_l2=overall_l2,
             dataset=DATASET, step=STEP)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_dims  = action_dim
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"SmolVLA × {DATASET}  |  step={STEP}  |  n={len(preds)} steps",
                 fontsize=12, fontweight="bold")

    # Bar chart per dimension
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, n_dims))
    bars = ax.bar(range(n_dims), per_dim_mae, color=colors, edgecolor="white")
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels(joint_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("MAE (original units)")
    ax.set_title("Per-Dimension Mean Absolute Error")
    ax.axhline(per_dim_mae.mean(), color="red", ls="--", lw=1,
               label=f"mean={per_dim_mae.mean():.4f}")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, per_dim_mae):
        ax.text(b.get_x() + b.get_width()/2, v + per_dim_mae.max()*0.02,
                f"{v:.3f}", ha="center", fontsize=8)

    # Trajectory (first 2 dims over time)
    ax2 = axes[1]
    n_show = min(300, len(preds))
    t_ax   = np.arange(n_show)
    colors2 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    for i in range(min(2, n_dims)):
        ax2.plot(t_ax, gts[:n_show, i],   color=colors2[i*2],   lw=1.5,
                 label=f"GT {joint_names[i]}")
        ax2.plot(t_ax, preds[:n_show, i], color=colors2[i*2+1], lw=1.5, ls="--",
                 label=f"Pred {joint_names[i]}")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Value (original units)")
    ax2.set_title("Predicted vs GT — first 2 dimensions")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / f"eval_{DATASET}_step{STEP}.png"
    plt.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
