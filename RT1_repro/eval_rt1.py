#!/usr/bin/env python
"""
RT-1 open-loop evaluation.

Same metric protocol as SmolVLA's eval_generic.py:
  - Per-dim MAE in original (unnormalized) action units
  - Overall L2 error per timestep
  Output:  RT1_repro/eval_output/eval_rt1_{DATASET}_step{STEP}.npz

Usage:
    DATASET=aloha_transfer STEP=3000 \
        /opt/anaconda3/envs/lerobot/bin/python RT1_repro/eval_rt1.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LEROBOT_SRC = "/Users/r/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dataset_configs import DATASET_CONFIGS
from train_generic import load_parquet_frames, decode_frame, get_task_description
from RT1_repro.train_rt1 import build_rt1, HISTORY_LEN, IMG_SIZE, ACTION_BINS

DATASET   = os.environ.get("DATASET", "aloha_transfer")
STEP      = int(os.environ.get("STEP", "3000"))
N_EVAL_EP = int(os.environ.get("N_EVAL_EP", "10"))
DATA_DIR  = Path(ROOT) / "Data"
CKPT_BASE = Path(ROOT) / "RT1_repro" / "checkpoints"
OUT_DIR   = Path(ROOT) / "RT1_repro" / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


def main():
    cfg    = DATASET_CONFIGS[DATASET]
    device = get_device()
    print(f"=== RT-1 eval: {DATASET}  step={STEP} ===")

    # Load checkpoint
    ckpt_path = CKPT_BASE / DATASET / f"step_{STEP:06d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    act_min = np.asarray(ckpt["act_min"], dtype=np.float32)
    act_max = np.asarray(ckpt["act_max"], dtype=np.float32)
    act_rng = act_max - act_min
    act_rng[act_rng < 1e-6] = 1.0

    model = build_rt1(cfg["action_dim"], device)
    # Re-freeze CLIP (matches train-time state)
    for name, p in model.named_parameters():
        if "conditioner" in name and "text_models" in name:
            p.requires_grad = False
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data
    local_names = cfg["local_name"] if isinstance(cfg["local_name"], list) else [cfg["local_name"]]
    dfs = []
    for name in local_names:
        ddir = DATA_DIR / name
        df = load_parquet_frames(ddir)
        df["_ddir"] = str(ddir)
        df["_task"] = get_task_description(ddir)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Pick last N_EVAL_EP episodes as held-out
    all_eps = sorted(df["episode_index"].unique())
    eval_eps = all_eps[-N_EVAL_EP:]
    print(f"  Evaluating on episodes {eval_eps[0]}..{eval_eps[-1]} ({len(eval_eps)} eps)")

    preds, gts = [], []

    with torch.no_grad():
        for ep in eval_eps:
            grp = df[df["episode_index"] == ep].reset_index(drop=True)
            ddir = Path(grp.iloc[0]["_ddir"])
            task = str(grp.iloc[0]["_task"])

            # Decode all frames for this episode, resized
            ep_frames = []
            for _, r in grp.iterrows():
                try:
                    img = decode_frame(ddir, int(r["index"]),
                                       cfg["image_key"], int(r["episode_index"]))
                except Exception:
                    img = np.zeros((cfg["image_h"], cfg["image_w"], 3), dtype=np.uint8)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                ep_frames.append(img)

            for t in range(HISTORY_LEN - 1, len(grp)):
                window = np.stack(ep_frames[t - HISTORY_LEN + 1: t + 1])  # (T,H,W,3)
                video  = torch.from_numpy(window).float() / 255.0
                video  = video.permute(3, 0, 1, 2).unsqueeze(0).to(device)

                logits = model(video, [task], cond_drop_prob=0.0)
                # logits: (1, T, action_dim, ACTION_BINS) — take last frame
                pred_bins = logits[0, -1].argmax(dim=-1).cpu().numpy()       # (action_dim,)
                pred_cont = (pred_bins / (ACTION_BINS - 1)) * act_rng + act_min

                gt = np.array(grp.iloc[t][cfg["action_key"]], dtype=np.float32)
                preds.append(pred_cont)
                gts.append(gt)

            print(f"    ep {ep}: {len(grp)} frames")

    preds = np.stack(preds)
    gts   = np.stack(gts)

    per_dim_mae = np.mean(np.abs(preds - gts), axis=0)
    overall_l2  = np.mean(np.linalg.norm(preds - gts, axis=1))

    print(f"\n  Overall L2: {overall_l2:.4f}")
    print(f"  Per-dim MAE: {per_dim_mae.round(3).tolist()}")

    out_npz = OUT_DIR / f"eval_rt1_{DATASET}_step{STEP}.npz"
    np.savez(out_npz,
             per_dim_mae=per_dim_mae,
             overall_l2=overall_l2,
             preds=preds, gts=gts,
             dataset=DATASET, step=STEP, model="rt1")
    print(f"  Saved: {out_npz}")

    # Per-dim bar
    fig, ax = plt.subplots(figsize=(max(6, len(per_dim_mae) * 0.6), 4))
    ax.bar(range(len(per_dim_mae)), per_dim_mae, color="#ff7f0e", edgecolor="white")
    ax.set_xlabel("Action dim")
    ax.set_ylabel("MAE (original units)")
    ax.set_title(f"RT-1 × {DATASET}  step={STEP}  L2={overall_l2:.3f}")
    ax.grid(True, axis="y", alpha=0.3)
    out_png = OUT_DIR / f"eval_rt1_{DATASET}_step{STEP}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
