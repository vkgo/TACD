---
license: other
license_name: tencent-hy-motion-1.0-community
license_link: LICENSE
library_name: transformers
tags:
  - text-to-motion
  - motion-generation
  - distillation
  - few-step
  - hy-motion
  - arxiv:2610.02867
base_model:
  - tencent/HY-Motion-1.0
extra_gated_prompt: >-
  This model is derived from Tencent HY-MOTION 1.0 and is released under the Tencent HY-MOTION 1.0
  Community License Agreement (see LICENSE).
extra_gated_fields:
  I agree to the license in LICENSE, including its Acceptable Use Policy and territory terms: checkbox
---

# TACD · HY-Motion-Lite, 8 steps

Text-to-motion in 8 sampling steps, distilled from HY-Motion 1.0.

📄 Paper: [https://arxiv.org/abs/2610.02867](https://arxiv.org/abs/2610.02867)

🌐 Project page: [https://vkgo.github.io/TACD/](https://vkgo.github.io/TACD/)

💻 Code: [https://github.com/vkgo/TACD](https://github.com/vkgo/TACD)

- <u>**268 ms per motion and 2.62 GB peak GPU memory in bf16**</u>, vs 2077 ms and 17.63 GB for the
  50-step HY-Motion 1.0 teacher: <u>**7.7× faster, 6.7× less memory**</u> (RTX L40, batch 1)
- 465M HY-Motion-1.0-Lite denoiser with a Qwen3-0.6B text encoder
- Trained only from the teacher on the model's own samples, with no motion data

## Usage

```bash
pip install torch "transformers>=4.51" safetensors einops numpy ftfy
```

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("weijinhuang/TACD-HY-Motion-Lite", trust_remote_code=True)
model = model.to("cuda").eval()   # or .to("cuda", torch.bfloat16) for bf16

out = model.generate(["a person walks forward, turns around and waves with the right hand"],
                     duration=4.0, seed=0)   # up to 12 s, 30 fps
out.joints   # (1, 120, 22, 3) joint positions in metres
out.rot6d    # (1, 120, 22, 6) local joint rotations
out.transl   # (1, 120, 3)     root translation
```

The text encoders (`Qwen/Qwen3-0.6B`, `openai/clip-vit-large-patch14`) are downloaded on first use;
for local copies, call `model.load_text_encoders(llm_path=..., clip_path=...)`.

## Results

HumanML3D test set:

| | R@1 ↑ | R@3 ↑ | MM-Dist ↓ | FID ↓ |
|---|---|---|---|---|
| HY-Motion 1.0 teacher @ 8 steps | 0.378 | 0.652 | 4.225 | 3.250 |
| **TACD @ 8 steps** | **0.479** | **0.782** | **3.192** | **1.041** |

## License

Released under the Tencent HY-MOTION 1.0 Community License Agreement ([LICENSE](LICENSE)).
Tencent HY-MOTION 1.0 is licensed under the Tencent HY-MOTION 1.0 Community License Agreement,
Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent HY” are owned by
Tencent or its affiliate. Powered by Tencent HY.

## Citation

```bibtex
@article{huang2026tacd,
  title   = {TACD: Distilling Efficient Text-to-Motion Models via Terminal Amplification Control},
  author  = {Huang, Wei-Jin and Li, Yuan-Ming and Lin, Kun-Yu and Luo, Wang and Zhu, Yinlin and
             Yu, Yue and Ye, Shenghao and Yuan, Junbin and Hong, Fa-Ting and Zhang, Qing and Zheng, Wei-Shi},
  journal = {arXiv preprint arXiv:2610.02867},
  year    = {2026}
}
```
