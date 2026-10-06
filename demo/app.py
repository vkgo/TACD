"""Web demo: type a sentence, watch the motion in a 3D viewer.

    python demo/app.py                 # then open http://localhost:7860
    python demo/app.py --share         # also print a temporary public link

Layout, styling and the viewer follow the Tencent HY-Motion-1.0 Space (gradio_app.py); changed by the
TACD authors: the model is TACD (8 steps, loaded with transformers), and prompt rewriting, CFG and FBX
export are removed.
"""
import argparse
import json
import os
import random
import time
from typing import List

import gradio as gr
import torch
from transformers import AutoModel

from visualize import HERE, MODEL_ID, motion_html
from hy_viewer.utils.gradio_css import APP_CSS, FOOTER_MD, HEADER_BASE_MD, get_placeholder_html
from hy_viewer.utils.visualize_mesh_web import generate_static_html_content

EXAMPLE_DIR = "examples/pregenerated"
MAX_SAMPLES = 4

with open(os.path.join(HERE, "examples", "example_prompts.json"), encoding="utf-8") as f:
    EXAMPLE_PROMPTS = json.load(f)  # [[prompt, duration], ...]
with open(os.path.join(HERE, "examples", "gallery.json"), encoding="utf-8") as f:
    EXAMPLE_GALLERY_LIST = json.load(f)  # [{prompt, duration, seeds, filename}, ...]

model = None


def _iframe(html: str, height: str) -> str:
    escaped = html.replace('"', "&quot;")
    return (f'<iframe srcdoc="{escaped}" width="100%" height="{height}" '
            'style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"></iframe>')


def _parse_seeds(seeds_csv: str) -> List[int]:
    seeds = [int(s) for s in seeds_csv.replace(" ", "").split(",") if s != ""]
    if not seeds:
        raise ValueError("enter at least one seed, e.g. 0")
    if len(seeds) > MAX_SAMPLES:
        raise ValueError(f"at most {MAX_SAMPLES} seeds")
    return seeds


def generate_motion(text: str, seeds_csv: str, duration: float):
    text = (text or "").strip()
    if not text:
        return gr.update(), "⚠️ Please enter text first"
    try:
        seeds = _parse_seeds(seeds_csv)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate([text] * len(seeds), duration=float(duration), seed=seeds)
        elapsed = time.time() - t0
        html = motion_html(out, text, folder="output/gradio")
        n = len(seeds)
        return _iframe(html, "750px"), f"🎉 Generated {n} motion{'s' if n > 1 else ''} in {elapsed:.1f} s"
    except Exception as e:  # shown to the user instead of a stack trace
        print(f">>> Motion generation failed: {e}")
        return gr.update(), f"❌ Motion generation failed: {e}"


def gallery_html() -> str:
    items = []
    for ex in EXAMPLE_GALLERY_LIST:
        try:
            html = generate_static_html_content(folder_name=EXAMPLE_DIR, file_name=ex["filename"], hide_captions=False)
        except Exception as e:
            print(f">>> Failed to load example {ex['filename']}: {e}")
            continue
        items.append(f"""
            <div class="example-grid-item" style="background: var(--card-bg, #fff); border-radius: 12px;
                        padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary, #333);
                            margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {ex["prompt"]}
                </div>
                {_iframe(html, "350px")}
            </div>""")
    return f'<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; padding: 8px;">{"".join(items)}</div>'


def random_seed() -> str:
    return str(random.randint(0, 999))


def build_ui():
    choices = ["Custom Input"] + [p for p, _ in EXAMPLE_PROMPTS]
    durations = dict((p, d) for p, d in EXAMPLE_PROMPTS)
    with gr.Blocks(title="TACD: Text-to-Motion in 8 Steps") as demo:
        gr.Markdown(HEADER_BASE_MD, elem_classes=["main-header"])
        with gr.Row():
            with gr.Column(scale=2, elem_classes=["left-panel"]):
                text_input = gr.Textbox(
                    label="📝 Input Text",
                    placeholder="Describe one person's motion in English, e.g. \"a person walks forward, turns around "
                                "and waves\". Click [📚 Example Prompts] to see more examples.",
                    lines=3, max_lines=10, autoscroll=False)
                duration = gr.Slider(minimum=1.0, maximum=12.0, value=4.0, step=0.1,
                                     label="⏱️ Action Duration (seconds)", info="Feel free to adjust the action duration")
                generate_btn = gr.Button("🚀 Generate Motion", variant="primary", size="lg",
                                         elem_classes=["generate-button"])
                example_dropdown = gr.Dropdown(choices=choices, value="Custom Input", label="📚 Example Prompts",
                                               interactive=True)
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    with gr.Row():
                        seed_input = gr.Textbox(label="🎯 Random Seeds", value="0", scale=3,
                                                info=f"One motion per seed, up to {MAX_SAMPLES} (e.g. 0,1,2,3)")
                        dice_btn = gr.Button("🎲", variant="secondary", size="sm", scale=1, min_width=50)
                status = gr.Textbox(label="📊 Status Information", value="Enter your text and click [🚀 Generate Motion].",
                                    lines=1, max_lines=10, elem_classes=["status-textbox"])
            with gr.Column(scale=3):
                output_display = gr.HTML(value=get_placeholder_html(), show_label=False, elem_classes=["flask-display"])

        with gr.Accordion("🎬 Example Gallery", open=True):
            gr.HTML(value=gallery_html(), show_label=False, elem_classes=["example-gallery-display"])
            with gr.Row():
                use_btns = [gr.Button(f"📋 Use Example {i + 1}", variant="secondary", size="sm")
                            for i in range(len(EXAMPLE_GALLERY_LIST))]

        gr.Markdown(FOOTER_MD, elem_classes=["footer"])

        dice_btn.click(random_seed, outputs=[seed_input])
        example_dropdown.change(
            lambda c: (gr.update(), gr.update()) if c == "Custom Input" else (c, durations[c]),
            inputs=[example_dropdown], outputs=[text_input, duration])
        for i, btn in enumerate(use_btns):
            ex = EXAMPLE_GALLERY_LIST[i]
            btn.click(lambda ex=ex: (ex["prompt"], ex["seeds"], ex["duration"]),
                      outputs=[text_input, seed_input, duration])
        generate_btn.click(lambda: "Generating motion, please wait...", outputs=[status]).then(
            generate_motion, inputs=[text_input, seed_input, duration], outputs=[output_display, status])
    return demo


def main():
    global model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_ID, help="Hugging Face id or local copy of the model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    args = ap.parse_args()
    os.chdir(HERE)  # the viewer helpers resolve output folders against demo/

    print(f">>> Loading {args.model} on {args.device}")
    t = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    with torch.no_grad():  # the first call also builds the text encoders
        model.generate(["a person stands still"], duration=1.0, seed=0)
    print(f">>> Model ready in {time.time() - t:.1f} s")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, css=APP_CSS)


if __name__ == "__main__":
    main()
