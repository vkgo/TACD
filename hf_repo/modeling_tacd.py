"""TACD few-step text-to-motion student, loadable with
`transformers.AutoModel.from_pretrained(<repo>, trust_remote_code=True)`.

The student is a HY-Motion-1.0-Lite MMDiT denoiser whose 8B text encoder was replaced by
Qwen3-0.6B, distilled on-policy from the HY-Motion-1.0 teacher for an 8-step Euler sampler.

Parts of this file restate Tencent HY-Motion 1.0 code (the text-encoding logic of
hymotion/network/text_encoders/text_encoder.py and the classifier-free-guided Euler sampler);
see the notice in tacd_mmdit.py. Tencent HY-MOTION 1.0 is licensed under the Tencent HY-MOTION
1.0 Community License Agreement, Copyright (c) 2025 Tencent. All Rights Reserved.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_tacd import TACDConfig
from .tacd_mmdit import HunyuanMotionMMDiT


@dataclass
class TACDMotionOutput(ModelOutput):
    """
    latent:  (B, L, 201) normalised motion, as produced by the sampler (torch.Tensor).
    motion:  (B, L, 201) de-normalised motion: [0:3] root translation, [3:9] root rot6d,
             [9:135] rot6d of joints 1..21, [135:201] unused channels (np.ndarray, float64).
    joints:  (B, L, 22, 3) joint positions in metres from forward kinematics (np.ndarray, float32).
    rot6d:   (B, L, 22, 6) local joint rotations in 6D form (np.ndarray, float64).
    transl:  (B, L, 3) root translation (np.ndarray, float64).
    """

    latent: Optional[torch.Tensor] = None
    motion: Optional[np.ndarray] = None
    joints: Optional[np.ndarray] = None
    rot6d: Optional[np.ndarray] = None
    transl: Optional[np.ndarray] = None


def _rot6d_to_rotmat_np(rot6d: np.ndarray) -> np.ndarray:
    shape = rot6d.shape[:-1]
    x = rot6d.reshape(*shape, 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def _forward_kinematics(local_rotmats, translation, offsets, parents):
    T, J = local_rotmats.shape[:2]
    global_rotmats = np.zeros((T, J, 3, 3), dtype=np.float64)
    positions = np.zeros((T, J, 3), dtype=np.float64)
    global_rotmats[:, 0] = local_rotmats[:, 0]
    positions[:, 0] = translation
    for i in range(1, J):
        p = parents[i]
        global_rotmats[:, i] = global_rotmats[:, p] @ local_rotmats[:, i]
        offset_i = offsets[i].astype(np.float64)
        positions[:, i] = (global_rotmats[:, p] @ offset_i[..., None]).squeeze(-1) + positions[:, p]
    return positions.astype(np.float32)


class _TextEncoders:
    """Qwen3 (token features) + CLIP (pooled sentence feature), as in HY-Motion's HYTextModel.
    Kept outside the nn.Module tree so their weights are neither saved with TACD nor
    re-hosted: they are downloaded from their original repositories."""

    def __init__(self, config: TACDConfig, device, llm_path=None, clip_path=None, **hf_kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

        llm_path = llm_path or config.llm_name_or_path
        clip_path = clip_path or config.clip_name_or_path
        llm_rev = None if llm_path != config.llm_name_or_path else config.llm_revision
        clip_rev = None if clip_path != config.clip_name_or_path else config.clip_revision
        self.system_prompt = config.llm_system_prompt
        self.max_length_clip = config.max_length_clip
        self._orig_max_length_llm = config.max_length_llm

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            clip_path, revision=clip_rev, max_length=self.max_length_clip, **hf_kwargs)
        self.clip = CLIPTextModel.from_pretrained(clip_path, revision=clip_rev, **hf_kwargs)
        self.clip = self.clip.eval().requires_grad_(False).to(device)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, revision=llm_rev, padding_side="right", **hf_kwargs)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, revision=llm_rev, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, **hf_kwargs)
        self.llm = self.llm.eval().requires_grad_(False).to(device)
        self.crop_start = self._compute_crop_start()
        self.max_length_llm = self._orig_max_length_llm + self.crop_start
        self.device = torch.device(device)

    def _messages(self, text: str):
        return [{"role": "system", "content": f"{self.system_prompt}"}, {"role": "user", "content": f"{text}"}]

    def _compute_crop_start(self) -> int:
        marker = "<BOC>"
        s = self.llm_tokenizer.apply_chat_template(
            self._messages(marker), tokenize=False, add_generation_prompt=False, enable_thinking=False)
        full_ids = self.llm_tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = self.llm_tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        for i in range(0, len(full_ids) - len(marker_ids) + 1):
            if full_ids[i:i + len(marker_ids)] == marker_ids:
                return i
        return max(0, len(full_ids) - 1)

    @torch.no_grad()
    def encode(self, prompts: List[str]):
        device = self.device
        llm_text = [
            self.llm_tokenizer.apply_chat_template(
                self._messages(t), tokenize=False, add_generation_prompt=False, enable_thinking=False)
            for t in prompts
        ]
        enc = self.llm_tokenizer(
            llm_text, return_length=False, return_overflowing_tokens=False, truncation=True,
            return_attention_mask=True, max_length=self.max_length_llm, padding="max_length",
            return_tensors="pt")
        out = self.llm(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                       output_hidden_states=True)
        ctxt = out.hidden_states[-1].clone()
        start, end = self.crop_start, self.crop_start + self._orig_max_length_llm
        ctxt = ctxt[:, start:end].contiguous()
        ctxt_len = (enc["attention_mask"].sum(dim=-1).to(device) - start).clamp(min=0, max=self._orig_max_length_llm)

        cenc = self.clip_tokenizer(
            prompts, return_length=False, return_overflowing_tokens=False, truncation=True,
            return_attention_mask=True, max_length=self.max_length_clip, padding=True, return_tensors="pt")
        cout = self.clip(input_ids=cenc["input_ids"].to(device), attention_mask=cenc["attention_mask"].to(device))
        vtxt = cout.pooler_output.unsqueeze(1)
        return vtxt, ctxt, ctxt_len


class TACDPreTrainedModel(PreTrainedModel):
    config_class = TACDConfig
    base_model_prefix = "tacd"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]

    def _init_weights(self, module):
        # Every weight is loaded from model.safetensors; keep PyTorch's default init otherwise.
        pass


class TACDModel(TACDPreTrainedModel):
    """TACD student: MMDiT denoiser + null (unconditional) text features.

    Typical use:
        model = AutoModel.from_pretrained(repo, trust_remote_code=True).to("cuda")
        out = model.generate(["a person walks forward and waves"], num_frames=120, seed=0)
        out.joints  # (1, 120, 22, 3) at 30 fps
    """

    def __init__(self, config: TACDConfig):
        super().__init__(config)
        na = dict(config.network_args)
        self.transformer = HunyuanMotionMMDiT(**na)
        # Encoder swap: the student's 1024-d Qwen3-0.6B features are mapped to the teacher's
        # 4096-d context space (align), then through the Lite model's own context projection.
        self.transformer.ctxt_encoder = nn.Sequential(
            nn.Linear(config.text_encoder_dim, na["ctxt_input_dim"]),
            nn.Linear(na["ctxt_input_dim"], na["feat_dim"]),
        )
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, na["vtxt_input_dim"]))
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, config.text_encoder_dim))
        self.__dict__["_text_encoders"] = None
        self.post_init()
        # Inference release: parameters are frozen, as in the paper's generation path. This is not
        # only about memory: torch.matmul folds a non-contiguous 3-D input differently when the
        # weight requires grad, which changes fp32 results at the 1e-4 level. Call
        # `model.requires_grad_(True)` to fine-tune.
        self.requires_grad_(False)

    # ------------------------------------------------------------------ text
    def load_text_encoders(self, device=None, llm_path: Optional[str] = None,
                           clip_path: Optional[str] = None, **hf_kwargs):
        """Download / load Qwen3-0.6B and CLIP ViT-L/14 (pinned revisions from config.json).
        Pass local directories in `llm_path` / `clip_path` to run offline."""
        device = device or self.device
        self.__dict__["_text_encoders"] = _TextEncoders(self.config, device, llm_path, clip_path, **hf_kwargs)
        return self.__dict__["_text_encoders"]

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        te = self.__dict__["_text_encoders"] or self.load_text_encoders()
        vtxt, ctxt, ctxt_len = te.encode(list(prompts))
        dev = self.device
        vtxt, ctxt = vtxt.to(dev), ctxt.to(dev)
        ctxt_mask = torch.zeros(ctxt.shape[0], ctxt.shape[1], dtype=torch.bool, device=dev)
        for i, l in enumerate(ctxt_len):
            ctxt_mask[i, :l] = True
        dtype = self.dtype
        return {"vtxt_input": vtxt.to(dtype), "ctxt_input": ctxt.to(dtype), "ctxt_mask": ctxt_mask}

    # ------------------------------------------------------------------ denoiser
    def forward(self, x, timesteps, vtxt_input, ctxt_input, ctxt_mask, x_mask=None):
        """Velocity prediction without guidance. x: (B, L, 201) normalised motion,
        timesteps: (B,) flow time in [0, 1] (0 = noise, 1 = data)."""
        if x_mask is None:
            x_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return self.transformer(x=x, ctxt_input=ctxt_input, vtxt_input=vtxt_input, timesteps=timesteps,
                                x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)

    def velocity(self, x, t, text_cond, guidance_scale: float, x_mask=None):
        """Classifier-free-guided velocity, identical to the sampler used in the paper."""
        B = x.shape[0]
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[1], dtype=torch.bool, device=x.device)
        vtxt, ctxt, ctxt_mask = text_cond["vtxt_input"], text_cond["ctxt_input"], text_cond["ctxt_mask"]
        device_type = "cuda" if x.device.type == "cuda" else "cpu"
        model_dtype = next(self.transformer.parameters()).dtype
        if guidance_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            t_in = t.expand(B * 2)
            x_mask_in = torch.cat([x_mask, x_mask], dim=0)
            vtxt_in = torch.cat([self.null_vtxt_feat.expand(B, -1, -1), vtxt], dim=0)
            ctxt_in = torch.cat([self.null_ctxt_input.expand(*ctxt.shape), ctxt], dim=0)
            ctxt_mask_in = torch.cat([ctxt_mask, ctxt_mask], dim=0)
            with torch.amp.autocast(device_type, dtype=model_dtype):
                v = self.transformer(x=x_in, ctxt_input=ctxt_in, vtxt_input=vtxt_in, timesteps=t_in,
                                     x_mask_temporal=x_mask_in, ctxt_mask_temporal=ctxt_mask_in)
            v_uncond, v_cond = v.chunk(2, dim=0)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        with torch.amp.autocast(device_type, dtype=model_dtype):
            return self.transformer(x=x, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t.expand(B),
                                    x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, text_cond: Dict[str, torch.Tensor],
               num_inference_steps: Optional[int] = None, guidance_scale: Optional[float] = None):
        """Uniform-grid Euler integration from t=0 (noise) to t=1 (motion)."""
        K = num_inference_steps or self.config.num_inference_steps
        s = self.config.guidance_scale if guidance_scale is None else guidance_scale
        x = noise
        dt = 1.0 / K
        x_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        for i in range(K):
            t = torch.tensor([i * dt], device=x.device, dtype=x.dtype)
            v = self.velocity(x, t, text_cond, s, x_mask=x_mask)
            x = x + v * dt
        return x

    # ------------------------------------------------------------------ public API
    @staticmethod
    def make_noise(num_frames: int, seeds: Sequence[int], dim: int = 201) -> torch.Tensor:
        """One CPU generator per sample, so a sample's noise depends only on its own seed."""
        gen = torch.Generator(device="cpu")
        out = []
        for s in seeds:
            gen.manual_seed(int(s))
            out.append(torch.randn(num_frames, dim, generator=gen))
        return torch.stack(out, 0)

    @torch.no_grad()
    def generate(self, prompts: Union[str, Sequence[str]], num_frames: Optional[int] = None,
                 duration: Optional[float] = None, seed: Union[int, Sequence[int], None] = 0,
                 noise: Optional[torch.Tensor] = None, num_inference_steps: Optional[int] = None,
                 guidance_scale: Optional[float] = None, decode: bool = True) -> TACDMotionOutput:
        """Generate motions for a batch of prompts (one shared length per call).

        num_frames / duration: sequence length in frames at config.fps (30) or in seconds;
            at most config.max_frames (360 = 12 s). Default 120 frames.
        seed: int (sample i uses seed + i) or one seed per prompt. Ignored if `noise` is given.
        num_inference_steps: the student was distilled for 8 steps; other values are untested.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        B = len(prompts)
        if noise is None:
            if num_frames is None:
                num_frames = int(round(duration * self.config.fps)) if duration is not None else 120
            if not 1 <= num_frames <= self.config.max_frames:
                raise ValueError(f"num_frames must be in [1, {self.config.max_frames}], got {num_frames}")
            seeds = [seed + i for i in range(B)] if isinstance(seed, int) else list(seed)
            if len(seeds) != B:
                raise ValueError(f"got {len(seeds)} seeds for {B} prompts")
            noise = self.make_noise(num_frames, seeds, self.config.network_args["input_dim"])
        noise = noise.to(self.device, dtype=self.dtype)
        text_cond = self.encode_text(prompts)
        latent = self.sample(noise, text_cond, num_inference_steps, guidance_scale)
        out = TACDMotionOutput(latent=latent)
        if decode:
            out.motion, out.joints, out.rot6d, out.transl = self.decode(latent)
        return out

    def decode(self, latent: Union[torch.Tensor, np.ndarray]):
        """Normalised 201-d motion -> (motion, joints, rot6d, transl); same math as the paper's
        evaluation bridge (no smoothing, no ground alignment)."""
        m = latent.float().cpu().numpy() if torch.is_tensor(latent) else np.asarray(latent, dtype=np.float32)
        single = m.ndim == 2
        if single:
            m = m[None]
        mean = np.asarray(self.config.motion_mean, dtype=np.float64)
        std = np.asarray(self.config.motion_std, dtype=np.float64)
        tmpl = np.asarray(self.config.joint_template, dtype=np.float32)
        parents = self.config.parents
        offsets = np.zeros((22, 3), dtype=np.float32)
        for i in range(22):
            offsets[i] = tmpl[i] if parents[i] == -1 else tmpl[i] - tmpl[parents[i]]
        motions, joints, rot6ds, transls = [], [], [], []
        for b in range(m.shape[0]):
            mb = m[b].copy() * std + mean
            T = mb.shape[0]
            transl = mb[:, 0:3]
            rot6d = np.concatenate([mb[:, 3:9], mb[:, 9:135]], axis=-1).reshape(T, 22, 6)
            joints.append(_forward_kinematics(_rot6d_to_rotmat_np(rot6d), transl, offsets, parents))
            motions.append(mb); rot6ds.append(rot6d); transls.append(transl)
        res = [np.stack(x, 0) for x in (motions, joints, rot6ds, transls)]
        return tuple(r[0] for r in res) if single else tuple(res)
