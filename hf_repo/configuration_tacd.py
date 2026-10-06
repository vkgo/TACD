"""Configuration for TACD few-step text-to-motion students (transformers remote code)."""
from typing import List, Optional

from transformers import PretrainedConfig

# Denoiser hyper-parameters of HY-Motion-1.0-Lite (ckpts/tencent/HY-Motion-1.0-Lite/config.yml).
LITE_NETWORK_ARGS = {
    "input_dim": 201,
    "feat_dim": 1024,
    "ctxt_input_dim": 4096,
    "vtxt_input_dim": 768,
    "num_layers": 18,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "mask_mode": "narrowband",
    "apply_rope_to_single_branch": False,
    "time_factor": 1000.0,
}

# Kinematic tree of the first 22 joints of the HY-Motion skeleton.
PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


class TACDConfig(PretrainedConfig):
    """
    Args:
        network_args: keyword arguments of the MMDiT denoiser (HunyuanMotionMMDiT).
        text_encoder_dim: hidden size of the student's LLM text encoder. The denoiser's context
            projection is `Linear(text_encoder_dim -> ctxt_input_dim) -> Linear(ctxt_input_dim -> feat_dim)`.
        llm_name_or_path / llm_revision: the student's LLM text encoder (Qwen3-0.6B).
        clip_name_or_path / clip_revision: the pooled sentence encoder (CLIP ViT-L/14 text tower).
        max_length_llm / max_length_clip: token budgets, as in HY-Motion.
        llm_system_prompt: system prompt wrapped around every caption before LLM encoding.
        num_inference_steps: number of uniform Euler steps the student was distilled for.
        guidance_scale: classifier-free guidance scale used in training and evaluation.
        fps: frame rate of generated motion.
        max_frames: longest sequence seen in training (frames at `fps`).
        motion_mean / motion_std: per-channel statistics of the 201-d motion representation.
        joint_template: rest-pose positions of the first 22 skeleton joints (float32 values).
        parents: kinematic parent of each of the 22 joints.
    """

    model_type = "tacd"

    def __init__(
        self,
        network_args: Optional[dict] = None,
        text_encoder_dim: int = 1024,
        llm_name_or_path: str = "Qwen/Qwen3-0.6B",
        llm_revision: Optional[str] = None,
        clip_name_or_path: str = "openai/clip-vit-large-patch14",
        clip_revision: Optional[str] = None,
        max_length_llm: int = 128,
        max_length_clip: int = 77,
        llm_system_prompt: str = "",
        num_inference_steps: int = 8,
        guidance_scale: float = 5.0,
        fps: float = 30.0,
        max_frames: int = 360,
        motion_mean: Optional[List[float]] = None,
        motion_std: Optional[List[float]] = None,
        joint_template: Optional[List[List[float]]] = None,
        parents: Optional[List[int]] = None,
        **kwargs,
    ):
        self.network_args = dict(LITE_NETWORK_ARGS if network_args is None else network_args)
        self.text_encoder_dim = text_encoder_dim
        self.llm_name_or_path = llm_name_or_path
        self.llm_revision = llm_revision
        self.clip_name_or_path = clip_name_or_path
        self.clip_revision = clip_revision
        self.max_length_llm = max_length_llm
        self.max_length_clip = max_length_clip
        self.llm_system_prompt = llm_system_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.max_frames = max_frames
        self.motion_mean = motion_mean
        self.motion_std = motion_std
        self.joint_template = joint_template
        self.parents = list(PARENTS_22 if parents is None else parents)
        super().__init__(**kwargs)
