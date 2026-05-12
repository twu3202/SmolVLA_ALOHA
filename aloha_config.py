"""
ALOHA-specific SmolVLA configuration.

ALOHA is a bimanual robot with 7 DoF per arm (14 total).
The sim dataset provides one overhead camera and absolute joint positions.

Dataset: lerobot/aloha_sim_transfer_cube_human
Task   : "Pick up the cube with the right arm and transfer it to the left arm."

State vector (14-dim): left_joints(7) + right_joints(7)
Action vector (14-dim): left_joints(7) + right_joints(7)  [absolute positions]
Camera: observation.images.top (480×640, resized to 512×512)
"""

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

# ── Dataset feature keys ───────────────────────────────────────────────────────
ALOHA_IMAGE_KEY  = "observation.images.top"
ALOHA_STATE_KEY  = "observation.state"
ALOHA_ACTION_KEY = "action"

TASK_DESCRIPTION = (
    "Pick up the cube with the right arm and transfer it to the left arm."
)

STATE_DIM  = 14   # 7 left joints + 7 right joints
ACTION_DIM = 14   # same (absolute joint positions)

IMAGE_H, IMAGE_W = 480, 640    # native ALOHA sim resolution


def make_aloha_smolvla_config(device: str = "mps") -> SmolVLAConfig:
    """Return a SmolVLAConfig adapted for ALOHA sim."""
    return SmolVLAConfig(
        # ── Feature specification ──────────────────────────────────────────────
        input_features={
            ALOHA_STATE_KEY: PolicyFeature(
                type=FeatureType.STATE,
                shape=(STATE_DIM,),
            ),
            ALOHA_IMAGE_KEY: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMAGE_H, IMAGE_W),
            ),
        },
        output_features={
            ALOHA_ACTION_KEY: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(ACTION_DIM,),
            ),
        },
        # ── Normalization ──────────────────────────────────────────────────────
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE":  NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        # ── Architecture ───────────────────────────────────────────────────────
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        train_expert_only=False,       # train VLM + expert (14-dim is new)
        max_state_dim=32,
        max_action_dim=32,
        # ── Temporal / action chunking ─────────────────────────────────────────
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=50,
        num_steps=10,                  # flow-matching denoising steps
        # ── Image handling ────────────────────────────────────────────────────
        resize_imgs_with_padding=(512, 512),
        # ── Language ─────────────────────────────────────────────────────────
        tokenizer_max_length=48,
        # ── Device ───────────────────────────────────────────────────────────
        device=device,
    )
