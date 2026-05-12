#!/usr/bin/env python
"""
Generic SmolVLA training script for any LeRobot v3 parquet dataset.

Reads dataset config from dataset_configs.py; handles ALOHA (14-dim bimanual),
xArm (4-dim), and any other single-camera LeRobot dataset.

Usage:
    DATASET=aloha_insertion /opt/anaconda3/envs/lerobot/bin/python train_generic.py
    DATASET=xarm_lift       /opt/anaconda3/envs/lerobot/bin/python train_generic.py
    DATASET=xarm_push       /opt/anaconda3/envs/lerobot/bin/python train_generic.py
    DATASET=aloha_multitask /opt/anaconda3/envs/lerobot/bin/python train_generic.py

Env overrides (all optional):
    TRAIN_STEPS, BATCH_SIZE, LR, SAVE_EVERY, LOG_EVERY
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

LEROBOT_SRC = "/Users/r/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from dataset_configs import DATASET_CONFIGS

# ── Runtime config ────────────────────────────────────────────────────────────
DATASET    = os.environ.get("DATASET", "aloha_insertion")
DATA_DIR   = Path("./Data")
CKPT_BASE  = Path("./checkpoints")
LOG_EVERY  = int(os.environ.get("LOG_EVERY",  "50"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "500"))
# ──────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available():         return "cuda"
    return "cpu"


# ── Data helpers ──────────────────────────────────────────────────────────────

def ensure_downloaded(cfg: dict, data_dir: Path):
    """Download dataset(s) from HuggingFace if not already present."""
    repos   = cfg["hf_repo"]   if isinstance(cfg["hf_repo"],   list) else [cfg["hf_repo"]]
    names   = cfg["local_name"] if isinstance(cfg["local_name"], list) else [cfg["local_name"]]
    for repo, name in zip(repos, names):
        dest = data_dir / name
        if not dest.exists():
            print(f"  Downloading {repo} → {dest} ...")
            snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=str(dest),
                              ignore_patterns=["*.gitattributes"])
            print(f"  Done.")
        else:
            print(f"  Found: {dest}")


def load_parquet_frames(dataset_dir: Path) -> pd.DataFrame:
    dfs = []
    for pq in sorted((dataset_dir / "data").rglob("*.parquet")):
        dfs.append(pd.read_parquet(pq))
    return pd.concat(dfs, ignore_index=True)


def load_stats(dataset_dir: Path) -> dict:
    with open(dataset_dir / "meta" / "stats.json") as f:
        return json.load(f)


def get_task_description(dataset_dir: Path) -> str:
    tp = dataset_dir / "meta" / "tasks.parquet"
    if tp.exists():
        df = pd.read_parquet(tp)
        if len(df) > 0:
            return df.index[0]
    return "Perform the robot manipulation task."


def decode_frame(dataset_dir: Path, global_index: int,
                 image_key: str, episode_index: int = 0) -> np.ndarray:
    """Read one frame from the dataset MP4 using global frame index."""
    chunk_idx = episode_index // 1000
    video_dir = dataset_dir / "videos" / image_key
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    mp4_files = sorted(chunk_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 in {chunk_dir}")
    cap = cv2.VideoCapture(str(mp4_files[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {global_index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class LeRobotDataset(Dataset):
    """
    Generic LeRobot v3 parquet dataset for SmolVLA training.
    Handles any single-camera dataset described in DATASET_CONFIGS.
    """

    def __init__(self, cfg: dict, data_dir: Path, chunk_size: int):
        self.cfg        = cfg
        self.chunk_size = chunk_size
        self.image_key  = cfg["image_key"]

        local_names = cfg["local_name"] if isinstance(cfg["local_name"], list) \
                      else [cfg["local_name"]]
        all_dfs = []

        # Accumulate stats across datasets (use first dataset's stats as base,
        # sufficient for single-dataset runs; multitask uses transfer stats)
        self.state_mean = None
        self.state_std  = None
        self.act_mean   = None
        self.act_std    = None

        for name in local_names:
            ddir  = data_dir / name
            df    = load_parquet_frames(ddir)
            stats = load_stats(ddir)
            task  = get_task_description(ddir)

            if self.state_mean is None:
                self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
                self.state_std  = np.array(stats["observation.state"]["std"],  dtype=np.float32)
                self.act_mean   = np.array(stats["action"]["mean"], dtype=np.float32)
                self.act_std    = np.array(stats["action"]["std"],  dtype=np.float32)

            df["_ddir"] = str(ddir)
            df["_task"] = task
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)
        self.df = self.df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        self._valid_idx = self._build_valid_indices()
        total_ep = self.df["episode_index"].nunique()
        print(f"  {len(self.df)} frames | {total_ep} episodes | "
              f"{len(self._valid_idx)} valid start positions")

    def _build_valid_indices(self):
        valid = []
        for _, grp in self.df.groupby(["_ddir", "episode_index"]):
            idxs = grp.index.tolist()
            for i in range(len(idxs) - self.chunk_size + 1):
                valid.append(idxs[i])
        return valid

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, i):
        start_row = self._valid_idx[i]
        rows = self.df.iloc[start_row: start_row + self.chunk_size]
        obs_row = rows.iloc[0]
        ddir = Path(obs_row["_ddir"])

        # Image
        global_idx = int(obs_row["index"])
        ep_idx     = int(obs_row["episode_index"])
        try:
            img_np = decode_frame(ddir, global_idx, self.image_key, ep_idx)
        except Exception:
            h, w = self.cfg["image_h"], self.cfg["image_w"]
            img_np = np.zeros((h, w, 3), dtype=np.uint8)
        img = torch.from_numpy(img_np).float() / 255.0
        img = img.permute(2, 0, 1)                      # (3, H, W)

        # State
        s_raw = np.array(obs_row[self.cfg["state_key"]], dtype=np.float32)
        state = (s_raw - self.state_mean) / (self.state_std + 1e-8)

        # Action chunk
        acts = np.stack([
            np.array(r[self.cfg["action_key"]], dtype=np.float32)
            for _, r in rows.iterrows()
        ])                                               # (chunk, action_dim)
        acts_norm = (acts - self.act_mean) / (self.act_std + 1e-8)

        return {
            "image":   img,
            "state":   torch.from_numpy(state),
            "actions": torch.from_numpy(acts_norm),
            "task":    str(obs_row["_task"]),
        }


# ── SmolVLA config builder ────────────────────────────────────────────────────

def make_smolvla_config(cfg: dict, device: str) -> SmolVLAConfig:
    return SmolVLAConfig(
        input_features={
            cfg["state_key"]: PolicyFeature(
                type=FeatureType.STATE, shape=(cfg["state_dim"],)
            ),
            cfg["image_key"]: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, cfg["image_h"], cfg["image_w"]),
            ),
        },
        output_features={
            cfg["action_key"]: PolicyFeature(
                type=FeatureType.ACTION, shape=(cfg["action_dim"],)
            ),
        },
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=False,
        max_state_dim=32,
        max_action_dim=32,
        n_obs_steps=1,
        chunk_size=cfg["chunk_size"],
        n_action_steps=cfg["chunk_size"],
        num_steps=10,
        resize_imgs_with_padding=(512, 512),
        tokenizer_max_length=48,
        device=device,
    )


# ── Collate ───────────────────────────────────────────────────────────────────

def collate(items: list[dict], tokenizer, device: str, image_key: str,
            state_key: str, action_key: str) -> dict:
    images  = torch.stack([x["image"]   for x in items]).to(device)
    states  = torch.stack([x["state"]   for x in items]).to(device)
    actions = torch.stack([x["actions"] for x in items]).to(device)
    tasks   = [x["task"] for x in items]
    toks = tokenizer(
        [t + "\n" for t in tasks],
        return_tensors="pt", padding="max_length",
        max_length=48, truncation=True,
    )
    return {
        image_key:                   images,
        state_key:                   states,
        action_key:                  actions,
        OBS_LANGUAGE_TOKENS:         toks["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: toks["attention_mask"].bool().to(device),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = DATASET_CONFIGS[DATASET]
    device = get_device()

    TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", str(cfg["train_steps"])))
    BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  str(cfg["batch_size"])))
    LR          = float(os.environ.get("LR", "1e-4"))
    CKPT_DIR    = CKPT_BASE / DATASET
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== SmolVLA — {DATASET} ===")
    print(f"Steps={TRAIN_STEPS}  BS={BATCH_SIZE}  LR={LR}  Device={device}")
    print(f"Checkpoint: {CKPT_DIR}\n")

    # ── Download if needed ────────────────────────────────────────────────────
    print("Checking / downloading data ...")
    ensure_downloaded(cfg, DATA_DIR)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading dataset ...")
    ds = LeRobotDataset(cfg, DATA_DIR, chunk_size=cfg["chunk_size"])

    # Save stats for eval
    stats_out = {
        "observation.state": {
            "mean": ds.state_mean.tolist(), "std": ds.state_std.tolist()
        },
        "action": {
            "mean": ds.act_mean.tolist(), "std": ds.act_std.tolist()
        },
    }
    with open(CKPT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=lambda x: x,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding SmolVLA ...")
    model_cfg = make_smolvla_config(cfg, device)
    model = SmolVLAPolicy(model_cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total: {n_total:.1f}M  Trainable: {n_train:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=LR * 0.1,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training ...\n")
    model.train()
    data_iter   = iter(loader)
    loss_log    = []
    t0          = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        try:
            items = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            items = next(data_iter)

        batch = collate(items, tokenizer, device,
                        cfg["image_key"], cfg["state_key"], cfg["action_key"])
        optimizer.zero_grad()
        loss, _ = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        loss_log.append(loss.item())

        if step % LOG_EVERY == 0:
            avg   = np.mean(loss_log[-LOG_EVERY:])
            eta   = (TRAIN_STEPS - step) / (step / (time.time() - t0))
            print(f"  step {step:5d}/{TRAIN_STEPS} | "
                  f"loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"ETA {eta/60:.1f} min")
            sys.stdout.flush()

        if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
            ckpt_path = CKPT_DIR / f"step_{step:06d}.pt"
            torch.save({
                "step":         step,
                "loss":         float(np.mean(loss_log[-100:])),
                "policy_state": model.state_dict(),
                "loss_log":     loss_log,
                "dataset":      DATASET,
                "cfg":          cfg,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"\n=== Done: {DATASET} in {(time.time()-t0)/60:.1f} min ===")
    print(f"Final loss: {np.mean(loss_log[-100:]):.4f}")


if __name__ == "__main__":
    main()
