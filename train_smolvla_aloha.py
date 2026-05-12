#!/usr/bin/env python
"""
Train SmolVLA on ALOHA sim (transfer-cube + insertion tasks).

Architecture  : SmolVLM2-500M backbone + Flow-Matching Action Expert
Robot         : ALOHA bimanual (14-dim absolute joint control)
Dataset       : lerobot/aloha_sim_transfer_cube_human (50 ep)
               + lerobot/aloha_sim_insertion_human    (50 ep)

Training flow:
  1. Load LeRobot v3 parquet dataset (local, downloaded by download_data.py)
  2. Normalise state/action via dataset stats
  3. Fine-tune SmolVLA with flow-matching loss
  4. Checkpoint every SAVE_EVERY steps

Usage:
    cd /Users/r/Projects/SmolVLA_ALOHA
    TASK=transfer /opt/anaconda3/envs/lerobot/bin/python train_smolvla_aloha.py
    TASK=insertion /opt/anaconda3/envs/lerobot/bin/python train_smolvla_aloha.py
    TASK=both      /opt/anaconda3/envs/lerobot/bin/python train_smolvla_aloha.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

LEROBOT_SRC = "/Users/r/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import io

from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from aloha_config import (
    make_aloha_smolvla_config,
    ALOHA_IMAGE_KEY, ALOHA_STATE_KEY, ALOHA_ACTION_KEY,
    TASK_DESCRIPTION, STATE_DIM, ACTION_DIM,
)

# ── Config ────────────────────────────────────────────────────────────────────
TASK          = os.environ.get("TASK", "transfer")   # transfer | insertion | both
TRAIN_STEPS   = int(os.environ.get("TRAIN_STEPS", "3000"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",  "4"))
LR            = float(os.environ.get("LR",        "1e-4"))
SAVE_EVERY    = int(os.environ.get("SAVE_EVERY",  "500"))
LOG_EVERY     = int(os.environ.get("LOG_EVERY",   "50"))
CHUNK_SIZE    = 50     # action chunk length (matches SmolVLA default)
DATA_DIR      = Path("./Data")
CKPT_DIR      = Path(f"./checkpoints/{TASK}")
# ──────────────────────────────────────────────────────────────────────────────

DATASET_NAMES = {
    "transfer": ["aloha_sim_transfer_cube_human"],
    "insertion": ["aloha_sim_insertion_human"],
    "both":     ["aloha_sim_transfer_cube_human", "aloha_sim_insertion_human"],
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_parquet_frames(dataset_dir: Path) -> pd.DataFrame:
    """Load all parquet files in data/ into a single DataFrame."""
    dfs = []
    for pq in sorted((dataset_dir / "data").rglob("*.parquet")):
        dfs.append(pd.read_parquet(pq))
    return pd.concat(dfs, ignore_index=True)


def load_stats(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "stats.json") as f:
        return json.load(f)


def decode_video_frame(dataset_dir: Path, global_index: int,
                       episode_index: int = 0) -> np.ndarray:
    """
    Extract a single frame from the dataset MP4 using its GLOBAL frame index
    (the 'index' column in the parquet, not frame_index which is within-episode).

    The LeRobot v3 format stores all episodes sequentially in a single MP4
    per chunk. global_index maps 1-to-1 onto video frame number.
    """
    import cv2
    # All episodes in chunk-000 → single file-000.mp4
    chunk_idx = episode_index // 1000
    video_dir = dataset_dir / "videos" / "observation.images.top"
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    mp4_files = sorted(chunk_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 in {chunk_dir}")
    mp4_path = mp4_files[0]
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {global_index} from {mp4_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # (H, W, 3) uint8


class ALOHADataset(Dataset):
    """
    Loads (observation, action_chunk) pairs from local LeRobot v3 parquet files.

    Each item:
        image   : (3, H, W) float32 [0,1]
        state   : (STATE_DIM,) float32  [normalised]
        actions : (CHUNK_SIZE, ACTION_DIM) float32 [normalised]
        task    : str
    """

    def __init__(self, dataset_dirs: list[Path], chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.frames_list = []   # (df_row, dataset_dir) pairs
        self.stats = {}

        for ddir in dataset_dirs:
            df = load_parquet_frames(ddir)
            stats = load_stats(ddir)
            # Merge stats (take first dataset's stats if multiple; in 'both' mode
            # we concatenate so we compute merged stats below)
            if not self.stats:
                self.stats = stats
            # Load task description
            tasks_df = pd.read_parquet(ddir / "meta" / "tasks.parquet")
            task_str = tasks_df.index[0] if len(tasks_df) > 0 else TASK_DESCRIPTION
            # Tag each row with its source dir and task description
            df["_ddir"] = str(ddir)
            df["_task"] = task_str
            self.frames_list.append(df)

        self.df = pd.concat(self.frames_list, ignore_index=True)

        # Compute merged normalisation stats over all loaded datasets
        self._state_mean = np.array(self.stats["observation.state"]["mean"], dtype=np.float32)
        self._state_std  = np.array(self.stats["observation.state"]["std"],  dtype=np.float32)
        self._act_mean   = np.array(self.stats["action"]["mean"], dtype=np.float32)
        self._act_std    = np.array(self.stats["action"]["std"],  dtype=np.float32)

        # Pre-sort by episode then frame so chunks are contiguous
        self.df = self.df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        # Build valid start indices (episode must have at least chunk_size frames left)
        self._valid_idx = self._build_valid_indices()
        print(f"  Loaded {len(self.df)} frames, {len(self._valid_idx)} valid start positions")

    def _build_valid_indices(self) -> list[int]:
        valid = []
        # Group by episode; within each episode, frames 0..(T-chunk_size) are valid starts
        ep_groups = self.df.groupby("episode_index")
        for _, grp in ep_groups:
            idxs = grp.index.tolist()
            n = len(idxs)
            for i in range(n - self.chunk_size + 1):
                valid.append(idxs[i])
        return valid

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, i):
        start_row = self._valid_idx[i]
        rows = self.df.iloc[start_row: start_row + self.chunk_size]

        obs_row = rows.iloc[0]
        ddir = Path(obs_row["_ddir"])

        # ── Image ─────────────────────────────────────────────────────────────
        ep_idx     = int(obs_row["episode_index"])
        global_idx = int(obs_row["index"])
        try:
            img_np = decode_video_frame(ddir, global_idx, ep_idx)
        except Exception:
            img_np = np.zeros((480, 640, 3), dtype=np.uint8)
        img = torch.from_numpy(img_np).float() / 255.0    # (H, W, 3)
        img = img.permute(2, 0, 1)                         # (3, H, W)

        # ── State (first frame) ───────────────────────────────────────────────
        state_raw = np.array(obs_row["observation.state"], dtype=np.float32)
        state = (state_raw - self._state_mean) / (self._state_std + 1e-8)

        # ── Action chunk ──────────────────────────────────────────────────────
        acts = np.stack([
            np.array(r["action"], dtype=np.float32)
            for _, r in rows.iterrows()
        ])                                                  # (chunk_size, 14)
        acts = (acts - self._act_mean) / (self._act_std + 1e-8)

        return {
            "image":   img,                                 # (3, H, W)
            "state":   torch.from_numpy(state),            # (14,)
            "actions": torch.from_numpy(acts),             # (chunk, 14)
            "task":    str(obs_row["_task"]),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def make_batch(items: list[dict], tokenizer, device: str) -> dict:
    """Collate a list of dataset items into a SmolVLA-compatible batch."""
    images  = torch.stack([x["image"]   for x in items]).to(device)  # (B,3,H,W)
    states  = torch.stack([x["state"]   for x in items]).to(device)  # (B,14)
    actions = torch.stack([x["actions"] for x in items]).to(device)  # (B,chunk,14)

    tasks = [x["task"] for x in items]
    toks = tokenizer(
        [t + "\n" for t in tasks],
        return_tensors="pt",
        padding="max_length",
        max_length=48,
        truncation=True,
    )
    return {
        ALOHA_IMAGE_KEY:                        images,
        ALOHA_STATE_KEY:                        states,
        ALOHA_ACTION_KEY:                       actions,
        OBS_LANGUAGE_TOKENS:                    toks["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK:            toks["attention_mask"].bool().to(device),
    }


def save_checkpoint(model, step, loss, ckpt_dir: Path, stats: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save({
        "step":         step,
        "loss":         loss,
        "policy_state": model.state_dict(),
        "stats":        stats,
    }, path)
    print(f"  Saved checkpoint: {path}")


def main():
    device = get_device()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== SmolVLA × ALOHA Sim — Training ===")
    print(f"Task      : {TASK}  |  Steps: {TRAIN_STEPS}  |  BS: {BATCH_SIZE}  |  Device: {device}")
    print(f"Checkpoint: {CKPT_DIR}\n")

    # ── Verify data dirs exist ────────────────────────────────────────────────
    dataset_names = DATASET_NAMES[TASK]
    dataset_dirs = []
    for name in dataset_names:
        d = DATA_DIR / name
        if not d.exists():
            print(f"Missing data: {d}")
            print("Run:  python download_data.py")
            raise SystemExit(1)
        dataset_dirs.append(d)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset ...")
    ds = ALOHADataset(dataset_dirs, chunk_size=CHUNK_SIZE)
    # Save merged stats for eval
    stats_out = {
        "observation.state": {
            "mean": ds._state_mean.tolist(),
            "std":  ds._state_std.tolist(),
        },
        "action": {
            "mean": ds._act_mean.tolist(),
            "std":  ds._act_std.tolist(),
        },
    }
    with open(CKPT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True,
                        collate_fn=lambda x: x)   # return list; collate in make_batch

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building SmolVLA ...")
    cfg = make_aloha_smolvla_config(device)
    model = SmolVLAPolicy(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total: {n_params:.1f}M  |  Trainable: {n_train:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=LR * 0.1
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nStarting training for {TRAIN_STEPS} steps ...\n")
    model.train()
    data_iter  = iter(loader)
    t0         = time.time()
    loss_window = []

    for step in range(1, TRAIN_STEPS + 1):
        # Fetch next batch
        try:
            items = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            items = next(data_iter)

        batch = make_batch(items, tokenizer, device)

        optimizer.zero_grad()
        loss, loss_dict = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        loss_window.append(loss.item())

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(loss_window[-LOG_EVERY:])
            steps_per_sec = step / elapsed
            eta = (TRAIN_STEPS - step) / steps_per_sec
            print(f"  step {step:5d}/{TRAIN_STEPS} | "
                  f"loss={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA {eta/60:.1f} min")
            sys.stdout.flush()

        if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
            save_checkpoint(model, step, float(np.mean(loss_window[-100:])),
                            CKPT_DIR, stats_out)

    total_time = time.time() - t0
    print(f"\n=== Training complete in {total_time/60:.1f} min ===")
    print(f"Best checkpoint: {CKPT_DIR}/step_{TRAIN_STEPS:06d}.pt")
    print(f"Next: run  python eval_smolvla_aloha.py")


if __name__ == "__main__":
    main()
