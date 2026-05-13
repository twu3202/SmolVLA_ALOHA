#!/usr/bin/env python
"""
RT-1 (lucidrains PyTorch port) training for any LeRobot v3 dataset.

Reuses dataset_configs.py and the data-loading helpers from train_generic.py,
but adapts for RT-1's expected inputs:
  - 6 frames of history (vs SmolVLA's 1 frame)
  - Discretized actions (256 bins per dim, cross-entropy loss)
  - 224x224 image resize (RT-1 standard)

Usage:
    DATASET=aloha_transfer TRAIN_STEPS=3000 BATCH_SIZE=4 \
        /opt/anaconda3/envs/lerobot/bin/python RT1_repro/train_rt1.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

# Bring SmolVLA_ALOHA root onto path so we can reuse helpers
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEROBOT_SRC = "/Users/r/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from robotic_transformer_pytorch import RT1, MaxViT

from dataset_configs import DATASET_CONFIGS
from train_generic import (
    ensure_downloaded, load_parquet_frames, load_stats,
    get_task_description, decode_frame,
)

# ── Runtime config ────────────────────────────────────────────────────────────
DATASET    = os.environ.get("DATASET", "aloha_transfer")
DATA_DIR   = Path(ROOT) / "Data"
CKPT_BASE  = Path(ROOT) / "RT1_repro" / "checkpoints"
LOG_EVERY  = int(os.environ.get("LOG_EVERY",  "50"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
ACTION_BINS = 256
HISTORY_LEN = 6
IMG_SIZE    = 224
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


# ── RT-1 dataset wrapper ──────────────────────────────────────────────────────

class RT1Dataset(Dataset):
    """
    For each valid frame index, returns:
      video:    (3, HISTORY_LEN, 224, 224)
      action:   (HISTORY_LEN, action_dim)  -- discretized to [0, 255]
      task:     str
    """

    def __init__(self, cfg: dict, data_dir: Path):
        self.cfg       = cfg
        self.image_key = cfg["image_key"]
        self.history   = HISTORY_LEN

        local_names = cfg["local_name"] if isinstance(cfg["local_name"], list) \
                      else [cfg["local_name"]]
        all_dfs = []
        for name in local_names:
            ddir  = data_dir / name
            df    = load_parquet_frames(ddir)
            stats = load_stats(ddir)
            task  = get_task_description(ddir)

            if not hasattr(self, "act_min"):
                a_min = np.array(stats["action"]["min"], dtype=np.float32)
                a_max = np.array(stats["action"]["max"], dtype=np.float32)
                # Guard against degenerate dims (min == max)
                rng = a_max - a_min
                rng[rng < 1e-6] = 1.0
                self.act_min = a_min
                self.act_max = a_max
                self.act_rng = rng

            df["_ddir"] = str(ddir)
            df["_task"] = task
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)
        self.df = self.df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        # Valid start positions: each episode needs >= HISTORY_LEN frames
        self._valid_idx = []
        for _, grp in self.df.groupby(["_ddir", "episode_index"]):
            idxs = grp.index.tolist()
            for i in range(self.history - 1, len(idxs)):
                self._valid_idx.append(idxs[i])

        total_ep = self.df["episode_index"].nunique()
        print(f"  {len(self.df)} frames | {total_ep} episodes | "
              f"{len(self._valid_idx)} valid windows (history={self.history})")

    def __len__(self):
        return len(self._valid_idx)

    def _discretize(self, a: np.ndarray) -> np.ndarray:
        """[A, action_dim] continuous → [A, action_dim] int64 in [0, 255]."""
        norm = (a - self.act_min) / self.act_rng
        idx  = np.clip(np.round(norm * (ACTION_BINS - 1)), 0, ACTION_BINS - 1)
        return idx.astype(np.int64)

    def __getitem__(self, i):
        end_row = self._valid_idx[i]
        start   = end_row - self.history + 1
        rows    = self.df.iloc[start: end_row + 1]
        ddir    = Path(rows.iloc[-1]["_ddir"])

        # Decode HISTORY_LEN frames
        frames = []
        for _, r in rows.iterrows():
            gi = int(r["index"])
            ep = int(r["episode_index"])
            try:
                img = decode_frame(ddir, gi, self.image_key, ep)
            except Exception:
                img = np.zeros((self.cfg["image_h"], self.cfg["image_w"], 3), dtype=np.uint8)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            frames.append(img)
        video = np.stack(frames)                          # (T, H, W, 3)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(3, 0, 1, 2)                 # (3, T, H, W)

        # Actions for the HISTORY_LEN frames
        acts = np.stack([
            np.array(r[self.cfg["action_key"]], dtype=np.float32)
            for _, r in rows.iterrows()
        ])                                                # (T, action_dim)
        acts_disc = self._discretize(acts)                # (T, action_dim) int64

        return {
            "video":   video,
            "actions": torch.from_numpy(acts_disc),
            "task":    str(rows.iloc[-1]["_task"]),
        }


def collate(items: list[dict]) -> dict:
    return {
        "video":   torch.stack([x["video"]   for x in items]),
        "actions": torch.stack([x["actions"] for x in items]),
        "tasks":   [x["task"] for x in items],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def build_rt1(action_dim: int, device: str) -> RT1:
    vit = MaxViT(num_classes=1000, dim_conv_stem=64, dim=96, dim_head=32,
                 depth=(2, 2, 5, 2), window_size=7, mbconv_expansion_rate=4,
                 mbconv_shrinkage_rate=0.25, dropout=0.1)
    return RT1(vit=vit, num_actions=action_dim, action_bins=ACTION_BINS,
               depth=6, heads=8, dim_head=64, cond_drop_prob=0.2,
               conditioner_kwargs=dict(model_types="clip")).to(device)


def main():
    cfg    = DATASET_CONFIGS[DATASET]
    device = get_device()

    TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", str(cfg["train_steps"])))
    BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  str(cfg["batch_size"])))
    LR          = float(os.environ.get("LR", "1e-4"))
    CKPT_DIR    = CKPT_BASE / DATASET
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== RT-1 — {DATASET} ===")
    print(f"Steps={TRAIN_STEPS}  BS={BATCH_SIZE}  LR={LR}  Device={device}")
    print(f"Checkpoint: {CKPT_DIR}\n")

    print("Checking / downloading data ...")
    ensure_downloaded(cfg, DATA_DIR)

    print("\nLoading dataset ...")
    ds = RT1Dataset(cfg, DATA_DIR)

    # Save action discretization stats for eval
    with open(CKPT_DIR / "action_stats.json", "w") as f:
        json.dump({
            "min": ds.act_min.tolist(),
            "max": ds.act_max.tolist(),
            "bins": ACTION_BINS,
            "action_dim": cfg["action_dim"],
        }, f, indent=2)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True, collate_fn=collate)

    print("\nBuilding RT-1 ...")
    model = build_rt1(cfg["action_dim"], device)

    # Freeze CLIP text encoder (heaviest, no robot info)
    n_total_pre = sum(p.numel() for p in model.parameters()) / 1e6
    for name, p in model.named_parameters():
        if "conditioner" in name and "text_models" in name:
            p.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total: {n_total:.1f}M  Trainable: {n_train:.1f}M  (CLIP frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=LR * 0.1,
    )

    print(f"\nStarting training ...\n")
    model.train()
    data_iter = iter(loader)
    loss_log  = []
    t0        = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        video   = batch["video"].to(device)               # (B, 3, T, H, W)
        targets = batch["actions"].to(device)             # (B, T, action_dim) int64

        optimizer.zero_grad()
        logits = model(video, batch["tasks"], cond_drop_prob=0.2)
        # logits: (B, T, action_dim, ACTION_BINS)
        B, T, A, K = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, K),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        loss_log.append(loss.item())

        if step % LOG_EVERY == 0:
            avg = np.mean(loss_log[-LOG_EVERY:])
            eta = (TRAIN_STEPS - step) / (step / (time.time() - t0))
            print(f"  step {step:5d}/{TRAIN_STEPS} | "
                  f"loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA {eta/60:.1f} min")
            sys.stdout.flush()

        if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
            ckpt_path = CKPT_DIR / f"step_{step:06d}.pt"
            torch.save({
                "step":      step,
                "loss":      float(np.mean(loss_log[-100:])),
                "model":     model.state_dict(),
                "loss_log":  loss_log,
                "dataset":   DATASET,
                "cfg":       cfg,
                "act_min":   ds.act_min,
                "act_max":   ds.act_max,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"\n=== Done: {DATASET} in {(time.time()-t0)/60:.1f} min ===")
    print(f"Final loss: {np.mean(loss_log[-100:]):.4f}")


if __name__ == "__main__":
    main()
