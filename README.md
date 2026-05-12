# SmolVLA × ALOHA × xArm — Cross-Dataset Reproduction

Reproducing [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on **11 LeRobot v3 datasets** spanning three robot platforms — bimanual ALOHA (sim + real), single-arm xArm, and their variants — all on a Mac (Apple MPS, no NVIDIA GPU).

Companion repo to [SmolVLA_cl](https://github.com/twu3202/SmolVLA_cl) (LIBERO + EEG modality experiments). Where the `cl` repo asks *"can we add a new modality?"*, this repo asks *"how well does SmolVLA generalise across embodiments and data distributions?"*

---

## TL;DR

| Rank | Dataset | L2 ↓ | MAE ↓ | Notes |
|---|---|---|---|---|
| 🥇 | `xarm_push` | **0.216** | 0.105 | Simplest task — 3-DOF position, no gripper |
| 🥈 | `aloha_static_battery` ★REAL | **0.651** | 0.121 | **Real-robot data beats sim** |
| 🥉 | `xarm_push_replay` | 0.529 | 0.269 | Replay variant |
| 4 | `aloha_insertion` | 0.716 | 0.154 | Sim, human demos |
| 5 | `aloha_transfer` | 0.729 | 0.149 | Sim, human demos |
| 6 | `aloha_static_coffee` ★REAL | 0.802 | 0.168 | Real-robot |
| 7 | `aloha_insertion_scripted` | 0.864 | 0.186 | Low train loss (0.040), higher L2 |
| 8 | `aloha_transfer_scripted` | 1.156 | 0.214 | Scripted |
| 9 | `xarm_lift_replay` | 1.042 | 0.448 | 84×84 imgs, grip struggles |
| 10 | `xarm_lift` | 1.324 | 0.565 | ❌ 84×84 resolution bottleneck |
| 11 | `aloha_multitask` | 1.327 | 0.276 | 2-task shared repr. penalty |

![Full cross-dataset comparison](eval_output/comparison_all_datasets.png)

---

## What this shows

Five empirical findings, each backed by at least one pair of runs:

| Finding | Evidence |
|---|---|
| **Real-robot data is no harder than sim** | `aloha_static_battery` L2=0.651 beats `aloha_transfer` L2=0.729 |
| **Scripted demos overfit differently** | `aloha_insertion_scripted`: 3× lower train loss (0.040) but *higher* open-loop L2 (0.864 vs 0.716) than human |
| **Image resolution matters for grasping** | `xarm_lift` (84×84) L2=1.32 vs `xarm_push` (no gripper) L2=0.22 — same image size, vastly different difficulty |
| **Replay distributions help marginally** | `xarm_lift` 1.32 → `xarm_lift_replay` 1.04 (lift), `xarm_push` 0.22 → replay 0.53 (push got worse) |
| **Multi-task pays ~2× penalty** | `aloha_multitask` L2=1.33 vs single-task ~0.72, even at 5000 steps |

---

## The 11 datasets

All datasets are in **LeRobot v3 format** (Parquet files + MP4 videos), freely available on HuggingFace under the `lerobot/` organisation.

### ALOHA bimanual robot — 14-dim joint space, 480×640 overhead camera

| Group | Dataset | HF Repo | Episodes |
|---|---|---|---|
| Sim, human demos | `aloha_transfer` | `lerobot/aloha_sim_transfer_cube_human` | 50 |
| Sim, human demos | `aloha_insertion` | `lerobot/aloha_sim_insertion_human` | 50 |
| Sim, human demos | `aloha_multitask` | both of the above | 100 |
| Sim, scripted | `aloha_transfer_scripted` | `lerobot/aloha_sim_transfer_cube_scripted` | 50 |
| Sim, scripted | `aloha_insertion_scripted` | `lerobot/aloha_sim_insertion_scripted` | 50 |
| ★ Real robot | `aloha_static_coffee` | `lerobot/aloha_static_coffee` | ~50 |
| ★ Real robot | `aloha_static_battery` | `lerobot/aloha_static_battery` | ~50 |

### xArm — single-arm robot, 84×84 images, 3-4 DOF action

| Group | Dataset | HF Repo | Action dim |
|---|---|---|---|
| Lift task | `xarm_lift` | `lerobot/xarm_lift_medium` | 4 (x,y,z,gripper) |
| Push task | `xarm_push` | `lerobot/xarm_push_medium` | 3 (x,y,z) |
| Lift replay | `xarm_lift_replay` | `lerobot/xarm_lift_medium_replay` | 4 |
| Push replay | `xarm_push_replay` | `lerobot/xarm_push_medium_replay` | 3 |

Full configs live in [`dataset_configs.py`](dataset_configs.py).

---

## Input / Output layout

### ALOHA Static (real robot) — `aloha_static_coffee`, `aloha_static_battery`

```
INPUT
  observation.images.cam_high   (3, 480, 640)   ← overhead RGB, normalized to [0,1]
  observation.state             (14,)           ← 14-dim joint positions, z-score normalized
  OBS_LANGUAGE_TOKENS           (48,)           ← tokenized task description
                                                  e.g. "Place the coffee capsule inside..."

OUTPUT
  action chunk                  (50, 14)        ← future 1 second of joint targets @ 50 FPS
                                                  z-score normalized; ×std+mean at inference
```

### xArm — `xarm_lift`, `xarm_push`

```
INPUT
  observation.image             (3, 84, 84)     ← single camera
  observation.state             (4,)            ← (x, y, z, gripper)
  OBS_LANGUAGE_TOKENS           (48,)           ← e.g. "Lift the red cube"

OUTPUT
  action chunk                  (10, 4 or 3)    ← future 0.67 s @ 15 FPS
```

### Internal SmolVLA processing

```
image → SmolVLM2 vision encoder (resize to 512×512, frozen)
                        │
                        ▼
language tokens → SmolVLM2 transformer → context tokens
                        │
state ──────────────────┼──→ Flow-matching action expert (~100M params, trained)
                        ▼
                10-step denoising → action chunk
```

---

## Quickstart

```bash
cd /Users/r/Projects/SmolVLA_ALOHA

# All scripts use the lerobot conda env:
PY=/opt/anaconda3/envs/lerobot/bin/python

# ── 1) Train one dataset (auto-downloads to ./Data on first run) ──────────────
DATASET=aloha_transfer       TRAIN_STEPS=3000 BATCH_SIZE=4 $PY train_generic.py
DATASET=aloha_static_coffee  TRAIN_STEPS=3000 BATCH_SIZE=4 $PY train_generic.py  # real-robot
DATASET=xarm_push            TRAIN_STEPS=3000 BATCH_SIZE=8 $PY train_generic.py
DATASET=aloha_multitask      TRAIN_STEPS=5000 BATCH_SIZE=4 $PY train_generic.py

# ── 2) Open-loop evaluation (no simulator needed) ────────────────────────────
DATASET=aloha_transfer       STEP=3000 N_EVAL_EP=10 $PY eval_generic.py
DATASET=aloha_static_coffee  STEP=3000 N_EVAL_EP=10 $PY eval_generic.py

# ── 3) Cross-dataset comparison plot (once all evals are done) ───────────────
$PY plot_comparison.py
```

Any dataset key in [`dataset_configs.py`](dataset_configs.py) is valid. To add a new LeRobot v3 dataset, append an entry to that registry and the generic train/eval scripts will handle it automatically.

---

## Files

| File | Purpose |
|---|---|
| `dataset_configs.py` | Registry of 13 dataset configurations — image keys, state/action dims, chunk size, FPS, train_steps, batch_size |
| `train_generic.py` | Generic trainer for any dataset in the registry — handles parquet loading, MP4 video decoding by global frame index, normalization stats, action-chunking, and flow-matching loss |
| `eval_generic.py` | Open-loop evaluation: feeds real observations frame-by-frame, calls `model.select_action()` each step, computes per-dim MAE + overall L2 in original units; saves `eval_{dataset}_step{step}.npz` and a per-dim bar chart |
| `plot_comparison.py` | Reads all `eval_*.npz` and produces the 4-panel cross-dataset figure (L2 bar, MAE bar, normalised heatmap, loss curves) |
| `download_data.py` | Stand-alone HuggingFace dataset downloader (rarely needed — training auto-downloads) |
| `aloha_config.py` | Legacy ALOHA-specific config — superseded by the registry in `dataset_configs.py` but kept for reference |
| `train_smolvla_aloha.py` | Legacy ALOHA-only trainer; superseded by `train_generic.py` |
| `eval_smolvla_aloha.py` | Legacy ALOHA-only evaluator; superseded by `eval_generic.py` |

---

## Environment

Apple MPS on M-series Mac; reuses the `lerobot` conda env from the parent project:

```bash
# Same env as SmolVLA_cl
PYTHONPATH is NOT required for this repo (we don't depend on LIBERO).
```

Dependencies are inherited from a local `lerobot` source install at `/Users/r/lerobot/src` (set via `sys.path.insert` in `train_generic.py`). If you don't have that, swap it for a `pip install lerobot` and adjust the path.

```bash
pip install lerobot transformers torch pandas pyarrow opencv-python matplotlib huggingface_hub
```

---

## Performance on M5 MPS

SmolVLA uses **action chunking** — one VLM forward pass generates 10–50 actions, executed one per step.

| Run | Steps | Time | Final loss |
|---|---|---|---|
| `aloha_transfer` (sim, human) | 3000 | ~28 min | 0.118 |
| `aloha_insertion_scripted` | 3000 | ~30 min | **0.040** |
| `aloha_transfer_scripted` | 3000 | ~30 min | 0.060 |
| `aloha_multitask` | 5000 | ~50 min | 0.456 |
| `aloha_static_coffee` ★REAL | 3000 | ~35 min | 0.110 |
| `aloha_static_battery` ★REAL | 3000 | ~33 min | 0.115 |
| `xarm_lift` | 3000 | ~20 min | 1.502 |
| `xarm_push` | 3000 | ~47 min | 0.300 |
| `xarm_lift_replay` | 3000 | ~48 min | 1.513 |
| `xarm_push_replay` | 3000 | ~48 min | 1.457 |

(Full eval results in [`eval_output/`](eval_output/); training checkpoints not committed — ~63 GB total across all runs.)

---

## Why these results, in one paragraph each

**`xarm_push` wins (L2=0.216)** — 3-DOF pure positional control with no gripper. The simplest task in the lineup. SmolVLA's flow-matching head fits it almost perfectly.

**`aloha_static_battery` beats sim ALOHA (L2=0.651 vs 0.729)** — Real-robot data is *not* harder than simulation for SmolVLA. The model handles the additional visual complexity (real lighting, real textures) well, and battery insertion is a relatively repeatable task.

**Scripted demos: lower train loss, higher open-loop L2** — `aloha_insertion_scripted` reaches loss 0.040 (3× lower than human 0.118) because scripted trajectories are highly consistent. But the model memorises those exact trajectories and fails to generalise to the slight variations in held-out episodes, giving worse open-loop L2 (0.864 vs 0.716).

**`xarm_lift` fails (L2=1.32)** — 84×84 images are too low-resolution to time the gripper open/close action precisely. The position trajectory is fine but the binary gripper signal is consistently mistimed. Confirmed by `xarm_push` (same 84×84 images, no gripper) being the *best* result in the entire benchmark.

**`aloha_multitask` pays a ~2× penalty (L2=1.33 vs ~0.72 single-task)** — Sharing one set of weights between transfer-cube and peg-insertion produces a representation that's mediocre at both. Even doubling the training budget to 5000 steps doesn't fully recover.

---

## Reproducing the overnight run

The shell script that produced the 11-dataset comparison ran sequentially over ~5.5 hours on an M5 MacBook:

```bash
DATASETS=(aloha_transfer aloha_insertion aloha_multitask
          aloha_transfer_scripted aloha_insertion_scripted
          aloha_static_coffee aloha_static_battery
          xarm_lift xarm_push xarm_lift_replay xarm_push_replay)

for d in "${DATASETS[@]}"; do
    DATASET=$d $PY train_generic.py
    DATASET=$d STEP=3000 $PY eval_generic.py
done
$PY plot_comparison.py
```

Training auto-downloads any missing dataset on first run. `Data/` is **not** committed (~2.6 GB) and `checkpoints/` is not committed (~63 GB) — both are listed in `.gitignore`.

---

## Related project

[**SmolVLA_cl**](https://github.com/twu3202/SmolVLA_cl) — SmolVLA + LIBERO + EEG as a fourth modality. Where this repo studies cross-embodiment generalisation, the `cl` repo studies whether a brain signal (EEG motor imagery) can be added as a controllable input. Both run on the same hardware (Apple MPS) and share the same SmolVLA architecture; only the embodiments and modalities differ.
