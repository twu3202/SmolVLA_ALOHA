"""
Per-dataset configuration registry.

Each entry specifies everything the generic training/eval scripts need
to handle a LeRobot v3 parquet dataset with SmolVLA.
"""

# ── Registry ──────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {

    # ── ALOHA sim: transfer cube ───────────────────────────────────────────────
    "aloha_transfer": {
        "hf_repo":     "lerobot/aloha_sim_transfer_cube_human",
        "local_name":  "aloha_sim_transfer_cube_human",
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA sim: peg insertion ──────────────────────────────────────────────
    "aloha_insertion": {
        "hf_repo":     "lerobot/aloha_sim_insertion_human",
        "local_name":  "aloha_sim_insertion_human",
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA sim: both tasks (multi-task) ────────────────────────────────────
    "aloha_multitask": {
        "hf_repo":     ["lerobot/aloha_sim_transfer_cube_human",
                        "lerobot/aloha_sim_insertion_human"],
        "local_name":  ["aloha_sim_transfer_cube_human",
                        "aloha_sim_insertion_human"],
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 5000,
        "batch_size":  4,
    },

    # ── xArm: lift cube ───────────────────────────────────────────────────────
    "xarm_lift": {
        "hf_repo":     "lerobot/xarm_lift_medium",
        "local_name":  "xarm_lift_medium",
        "image_key":   "observation.image",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   4,
        "action_dim":  4,
        "image_h":     84,
        "image_w":     84,
        "chunk_size":  10,
        "fps":         15,
        "train_steps": 5000,
        "batch_size":  8,
    },

    # ── xArm: push cube ──────────────────────────────────────────────────────
    "xarm_push": {
        "hf_repo":     "lerobot/xarm_push_medium",
        "local_name":  "xarm_push_medium",
        "image_key":   "observation.image",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   4,
        "action_dim":  3,
        "image_h":     84,
        "image_w":     84,
        "chunk_size":  10,
        "fps":         15,
        "train_steps": 5000,
        "batch_size":  8,
    },

    # ── ALOHA sim: insertion scripted (vs human comparison) ───────────────────
    "aloha_insertion_scripted": {
        "hf_repo":     "lerobot/aloha_sim_insertion_scripted",
        "local_name":  "aloha_sim_insertion_scripted",
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA sim: transfer scripted (vs human comparison) ────────────────────
    "aloha_transfer_scripted": {
        "hf_repo":     "lerobot/aloha_sim_transfer_cube_scripted",
        "local_name":  "aloha_sim_transfer_cube_scripted",
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA sim: scripted multi-task ────────────────────────────────────────
    "aloha_multitask_scripted": {
        "hf_repo":     ["lerobot/aloha_sim_transfer_cube_scripted",
                        "lerobot/aloha_sim_insertion_scripted"],
        "local_name":  ["aloha_sim_transfer_cube_scripted",
                        "aloha_sim_insertion_scripted"],
        "image_key":   "observation.images.top",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── xArm replay: lift (different data distribution) ──────────────────────
    "xarm_lift_replay": {
        "hf_repo":     "lerobot/xarm_lift_medium_replay",
        "local_name":  "xarm_lift_medium_replay",
        "image_key":   "observation.image",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   4,
        "action_dim":  4,
        "image_h":     84,
        "image_w":     84,
        "chunk_size":  10,
        "fps":         15,
        "train_steps": 3000,
        "batch_size":  8,
    },

    # ── xArm replay: push (different data distribution) ──────────────────────
    "xarm_push_replay": {
        "hf_repo":     "lerobot/xarm_push_medium_replay",
        "local_name":  "xarm_push_medium_replay",
        "image_key":   "observation.image",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   4,
        "action_dim":  3,
        "image_h":     84,
        "image_w":     84,
        "chunk_size":  10,
        "fps":         15,
        "train_steps": 3000,
        "batch_size":  8,
    },

    # ── ALOHA static: coffee (REAL ROBOT, overhead cam only) ─────────────────
    # 4 cameras available; use cam_high (overhead) to match sim setup
    "aloha_static_coffee": {
        "hf_repo":     "lerobot/aloha_static_coffee",
        "local_name":  "aloha_static_coffee",
        "image_key":   "observation.images.cam_high",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA static: battery insertion (REAL ROBOT) ─────────────────────────
    "aloha_static_battery": {
        "hf_repo":     "lerobot/aloha_static_battery",
        "local_name":  "aloha_static_battery",
        "image_key":   "observation.images.cam_high",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── PushT: 2-DOF planar pushing task ─────────────────────────────────────
    # Simplest possible manipulation: no gripper, 2D position control, 206 episodes
    # NOTE: parquet stores raw pixel coordinates (12-511), not z-scored values.
    # Set raw_gt=True so eval does NOT apply (gt * std + mean) to the parquet value.
    "pusht": {
        "hf_repo":     "lerobot/pusht",
        "local_name":  "pusht",
        "image_key":   "observation.image",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   2,
        "action_dim":  2,
        "image_h":     96,
        "image_w":     96,
        "chunk_size":  10,
        "fps":         10,
        "train_steps": 3000,
        "batch_size":  16,
        "raw_gt":      True,   # parquet stores raw pixel coords; skip denorm on GT
    },

    # ── ALOHA static: cups open (REAL ROBOT) ────────────────────────────────
    "aloha_static_cups_open": {
        "hf_repo":     "lerobot/aloha_static_cups_open",
        "local_name":  "aloha_static_cups_open",
        "image_key":   "observation.images.cam_high",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA static: towel fold (REAL ROBOT, deformable object) ─────────────
    "aloha_static_towel": {
        "hf_repo":     "lerobot/aloha_static_towel",
        "local_name":  "aloha_static_towel",
        "image_key":   "observation.images.cam_high",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA static: ziploc slide (REAL ROBOT, fine manipulation) ───────────
    "aloha_static_ziploc_slide": {
        "hf_repo":     "lerobot/aloha_static_ziploc_slide",
        "local_name":  "aloha_static_ziploc_slide",
        "image_key":   "observation.images.cam_high",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },

    # ── ALOHA static: coffee (dual camera — high + left_wrist) ───────────────
    # Tests whether adding a wrist camera improves real-robot performance
    "aloha_static_coffee_2cam": {
        "hf_repo":     "lerobot/aloha_static_coffee",
        "local_name":  "aloha_static_coffee",
        "image_key":   "observation.images.cam_high",
        "image_key2":  "observation.images.cam_left_wrist",
        "state_key":   "observation.state",
        "action_key":  "action",
        "state_dim":   14,
        "action_dim":  14,
        "image_h":     480,
        "image_w":     640,
        "chunk_size":  50,
        "fps":         50,
        "train_steps": 3000,
        "batch_size":  4,
    },
}
