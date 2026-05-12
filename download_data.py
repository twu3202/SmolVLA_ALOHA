#!/usr/bin/env python
"""
Download ALOHA sim dataset from HuggingFace into ./Data/

Downloads:
  - lerobot/aloha_sim_transfer_cube_human  (50 eps, ~500 MB)
  - lerobot/aloha_sim_insertion_human      (50 eps, ~300 MB)

Usage:
    /opt/anaconda3/envs/lerobot/bin/python download_data.py
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

DATA_DIR = Path("./Data")

DATASETS = [
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_insertion_human",
]


def download_dataset(repo_id: str, target_dir: Path):
    name = repo_id.split("/")[-1]
    dest = target_dir / name
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    print(f"  Downloading {repo_id} → {dest} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print(f"  Done: {dest}")
    return dest


def show_summary(dest: Path):
    info_path = dest / "meta" / "info.json"
    if not info_path.exists():
        return
    with open(info_path) as f:
        info = json.load(f)

    tasks_path = dest / "meta" / "tasks.parquet"
    task_str = "N/A"
    if tasks_path.exists():
        df = pd.read_parquet(tasks_path)
        task_str = df.index[0] if len(df) > 0 else "N/A"

    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"\n  {'='*55}")
    print(f"  Dataset    : {dest.name}")
    print(f"  Robot      : {info.get('robot_type', '?')}")
    print(f"  Episodes   : {info['total_episodes']}")
    print(f"  Frames     : {info['total_frames']}")
    print(f"  FPS        : {info['fps']}")
    print(f"  Task       : {task_str}")
    print(f"  Disk size  : {size_mb:.0f} MB")
    print(f"  {'='*55}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR.resolve()}\n")

    for repo_id in DATASETS:
        dest = download_dataset(repo_id, DATA_DIR)
        show_summary(dest)

    print("\nAll datasets ready.")
    print(f"Next: run  python train_smolvla_aloha.py")


if __name__ == "__main__":
    main()
