#!/usr/bin/env python
"""
Open-loop evaluation of SmolVLA on ALOHA sim.

Since we don't have the ALOHA MuJoCo simulator available, we run
**open-loop rollouts**: feed real observations from held-out episodes,
compare predicted action chunks to ground truth.

Metrics:
  - Action chunk L2 error (per dimension and overall)
  - Joint-position prediction error across time horizon
  - Per-arm breakdown (left vs right)
  - Loss curve plot

Usage:
    TASK=transfer STEP=3000 \
        /opt/anaconda3/envs/lerobot/bin/python eval_smolvla_aloha.py
"""

import os, sys, json, warnings, time
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from aloha_config import (
    make_aloha_smolvla_config,
    ALOHA_IMAGE_KEY, ALOHA_STATE_KEY, ALOHA_ACTION_KEY,
    TASK_DESCRIPTION, STATE_DIM, ACTION_DIM,
)
from train_smolvla_aloha import ALOHADataset, decode_video_frame, DATASET_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
TASK       = os.environ.get("TASK", "transfer")
STEP       = int(os.environ.get("STEP", "3000"))
N_EVAL_EP  = int(os.environ.get("N_EVAL_EP", "10"))   # episodes to evaluate
DATA_DIR   = Path("./Data")
CKPT_DIR   = Path(f"./checkpoints/{TASK}")
CKPT_PATH  = CKPT_DIR / f"step_{STEP:06d}.pt"
OUT_DIR    = Path("./eval_output")
# ──────────────────────────────────────────────────────────────────────────────

LEFT_JOINTS  = list(range(0, 7))    # indices in 14-dim vector
RIGHT_JOINTS = list(range(7, 14))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_obs_batch(img_np, state_raw, state_mean, state_std,
                    tokenizer, device, task_desc=None):
    """Package a single observation into a policy-compatible batch."""
    img = torch.from_numpy(img_np).float() / 255.0   # (H,W,3)
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    state = (state_raw.astype(np.float32) - state_mean) / (state_std + 1e-8)
    state = torch.from_numpy(state).unsqueeze(0).to(device)

    task = (task_desc or TASK_DESCRIPTION) + "\n"
    toks = tokenizer(task, return_tensors="pt", padding="max_length",
                     max_length=48, truncation=True)

    return {
        ALOHA_IMAGE_KEY:             img,
        ALOHA_STATE_KEY:             state,
        OBS_LANGUAGE_TOKENS:         toks["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: toks["attention_mask"].bool().to(device),
    }


def eval_episode(model, tokenizer, df_ep, dataset_dir, task_desc,
                 state_mean, state_std, act_mean, act_std, device):
    """
    Run step-by-step open-loop evaluation on one episode.

    select_action returns a single action per call (1, 14); it internally
    maintains a chunk queue and re-runs inference every n_action_steps steps.

    Returns arrays:
        preds : (T, 14) unnormalised predicted actions
        gts   : (T, 14) unnormalised ground-truth actions
    """
    model.reset()
    pred_list, gt_list = [], []
    T = len(df_ep)

    for t in range(T):
        row       = df_ep.iloc[t]
        ep_idx    = int(row["episode_index"])
        global_idx = int(row["index"])

        try:
            img_np = decode_video_frame(dataset_dir, global_idx, ep_idx)
        except Exception:
            img_np = np.zeros((480, 640, 3), dtype=np.uint8)

        state_raw = np.array(row["observation.state"], dtype=np.float32)
        batch = build_obs_batch(img_np, state_raw, state_mean, state_std,
                                tokenizer, device, task_desc=task_desc)

        # Returns (1, action_dim) – one step from internal action queue
        pred_norm = model.select_action(batch).squeeze(0).cpu().numpy()  # (14,)
        pred_list.append(pred_norm * act_std + act_mean)

        gt_norm = np.array(row["action"], dtype=np.float32)
        gt_list.append(gt_norm * act_std + act_mean)

    return np.stack(pred_list), np.stack(gt_list)   # (T,14), (T,14)


def main():
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== SmolVLA × ALOHA — Open-Loop Evaluation ===")
    print(f"Task: {TASK}  |  Step: {STEP}  |  Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}\n")

    if not CKPT_PATH.exists():
        print(f"Checkpoint not found: {CKPT_PATH}")
        print("Run training first:  python train_smolvla_aloha.py")
        raise SystemExit(1)

    # ── Load stats ─────────────────────────────────────────────────────────────
    stats_path = CKPT_DIR / "dataset_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
    state_std  = np.array(stats["observation.state"]["std"],  dtype=np.float32)
    act_mean   = np.array(stats["action"]["mean"], dtype=np.float32)
    act_std    = np.array(stats["action"]["std"],  dtype=np.float32)

    # ── Load model ─────────────────────────────────────────────────────────────
    cfg = make_aloha_smolvla_config(device)
    cfg.load_vlm_weights = False
    model = SmolVLAPolicy(cfg).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["policy_state"])
    model.eval()
    print(f"Loaded checkpoint step {ckpt['step']}  (train loss={ckpt['loss']:.4f})")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    # ── Load eval data ────────────────────────────────────────────────────────
    dataset_names = DATASET_NAMES[TASK]
    dataset_dirs  = [DATA_DIR / n for n in dataset_names]

    from train_smolvla_aloha import load_parquet_frames
    all_pairs = []
    for ddir in dataset_dirs:
        # Load task description from dataset metadata
        tasks_df = pd.read_parquet(ddir / "meta" / "tasks.parquet")
        task_desc = tasks_df.index[0] if len(tasks_df) > 0 else TASK_DESCRIPTION
        print(f"\n  Task: {task_desc}")

        df = load_parquet_frames(ddir)
        episodes = sorted(df["episode_index"].unique())
        eval_eps = episodes[-N_EVAL_EP:]    # last N episodes as eval set
        print(f"Evaluating {len(eval_eps)} episodes from {ddir.name} ...")

        for ep_idx in eval_eps:
            df_ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            ep_preds, ep_gts = eval_episode(
                model, tokenizer, df_ep, ddir, task_desc,
                state_mean, state_std, act_mean, act_std, device,
            )
            all_pairs.append((ep_preds, ep_gts))
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

    # ── Compute metrics ───────────────────────────────────────────────────────
    preds = np.concatenate([p for p, _ in all_pairs], axis=0)   # (N, 14)
    gts   = np.concatenate([g for _, g in all_pairs], axis=0)   # (N, 14)

    overall_l2  = np.sqrt(((preds - gts) ** 2).sum(axis=1)).mean()
    per_dim_mae = np.abs(preds - gts).mean(axis=0)               # (14,)
    left_mae    = per_dim_mae[LEFT_JOINTS].mean()
    right_mae   = per_dim_mae[RIGHT_JOINTS].mean()

    print(f"\n{'='*60}")
    print(f"Open-loop evaluation results  (n={len(preds)} chunks)")
    print(f"{'='*60}")
    print(f"  Overall chunk L2 error : {overall_l2:.4f} rad")
    print(f"  Left  arm MAE          : {left_mae:.4f} rad")
    print(f"  Right arm MAE          : {right_mae:.4f} rad")
    print(f"\n  Per-joint MAE:")
    joint_names = [
        "L_waist","L_shoulder","L_elbow","L_forearm","L_wrist_a","L_wrist_r","L_gripper",
        "R_waist","R_shoulder","R_elbow","R_forearm","R_wrist_a","R_wrist_r","R_gripper",
    ]
    for name, mae in zip(joint_names, per_dim_mae):
        bar = "█" * int(mae * 40)
        print(f"    {name:<14} {mae:.4f}  {bar}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"SmolVLA × ALOHA ({TASK}) — Open-Loop Action Prediction\n"
        f"step={STEP}  |  {len(preds)} eval chunks",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: per-joint MAE bar chart
    ax = axes[0]
    colors = ["#1f77b4"] * 7 + ["#ff7f0e"] * 7
    bars = ax.bar(range(14), per_dim_mae, color=colors, edgecolor="white")
    ax.set_xticks(range(14))
    ax.set_xticklabels(joint_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MAE (rad)")
    ax.set_title("Per-Joint Mean Absolute Error")
    ax.axhline(left_mae,  color="#1f77b4", ls="--", lw=1,
               label=f"Left arm mean {left_mae:.4f}")
    ax.axhline(right_mae, color="#ff7f0e", ls="--", lw=1,
               label=f"Right arm mean {right_mae:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (b, v) in enumerate(zip(bars, per_dim_mae)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002,
                f"{v:.3f}", ha="center", fontsize=7)

    # Panel 2: predicted vs GT for first joint over time
    ax2 = axes[1]
    n_show = min(200, len(preds))
    t_ax = np.arange(n_show)
    ax2.plot(t_ax, gts[:n_show, 0],   label="GT left_waist",   lw=1.5, alpha=0.8)
    ax2.plot(t_ax, preds[:n_show, 0], label="Pred left_waist", lw=1.5, ls="--", alpha=0.8)
    ax2.plot(t_ax, gts[:n_show, 7],   label="GT right_waist",  lw=1.5, alpha=0.8, color="#2ca02c")
    ax2.plot(t_ax, preds[:n_show, 7], label="Pred right_waist",lw=1.5, ls="--", alpha=0.8, color="#d62728")
    ax2.set_xlabel("Time step (across chunks)")
    ax2.set_ylabel("Joint position (rad)")
    ax2.set_title("Predicted vs Ground-Truth Trajectory (waist joints)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / f"eval_{TASK}_step{STEP}.png"
    plt.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")

    # Save numeric results
    np.savez(
        OUT_DIR / f"eval_{TASK}_step{STEP}.npz",
        preds=preds, gts=gts,
        per_dim_mae=per_dim_mae,
        overall_l2=overall_l2,
    )


if __name__ == "__main__":
    main()
