"""Render recorded controller states next to their reference; no policy is bypassed."""
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .paths import box_feet_xml, hy_motion_root


def sample_qpos(qpos, fps, times):
    source = np.arange(len(qpos)) / fps
    times = np.clip(times, source[0], source[-1])
    result = np.stack([np.interp(times, source, qpos[:, j]) for j in range(36)], axis=-1)
    result[:, 3:7] = Slerp(source, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(times).as_quat()[:, [3, 0, 1, 2]]
    return result


def render_comparison(reference, executed, executed_fps, out, fps=25, human=None, video_duration_seconds=None, post_motion_behavior="decode"):
    from .transition import generated_time
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(box_feet_xml()))
    model.vis.global_.offwidth = 640; model.vis.global_.offheight = 480
    data = mujoco.MjData(model)
    duration = (len(executed) - 1) / executed_fps
    recorded_duration = duration
    if video_duration_seconds is not None:
        duration = max(duration, video_duration_seconds)
    times = np.arange(0, duration + 1e-8, 1 / fps)
    actual = sample_qpos(executed, executed_fps, times)
    target = sample_qpos(reference.qpos, reference.fps, times)
    reference_end = (len(reference.qpos)-1)/reference.fps
    deploy_time = np.ceil(reference_end*50-1e-9)/50
    if post_motion_behavior == "deploy":
        target[times >= deploy_time] = reference.qpos[0]
    renderer = mujoco.Renderer(model, height=480, width=640)
    source = str(reference.metadata.get("retargeter", "umr")).upper()
    prefix = float(reference.metadata.get("stand_prefix_seconds", 0.))
    human_renderer = None
    if human is not None:
        human_model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="480"/></visual><worldbody><light pos="0 0 4"/><geom type="plane" size="10 10 .1" rgba=".8 .8 .8 1"/></worldbody></mujoco>')
        human_data = mujoco.MjData(human_model)
        mujoco.mj_forward(human_model, human_data)
        human_renderer = mujoco.Renderer(human_model, height=480, width=640)
        joints = human.joints_world[..., [2, 0, 1]]
        parents = np.fromfile(hy_motion_root() / "scripts/gradio/static/assets/dump_wooden/kintree.bin", dtype=np.int32)[:22]
        human_camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(human_camera)
        human_camera.distance = max(3., 1.8 * float(np.ptp(joints, axis=1).max()))
        human_camera.azimuth = 130; human_camera.elevation = -15
        human_center_z = float((joints[..., 2].min() + joints[..., 2].max()) / 2)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.6; camera.azimuth = 130; camera.elevation = -15
    try:
        with imageio.get_writer(out, fps=fps, codec="libx264", quality=8) as writer:
            for i, time in enumerate(times):
                images = []
                source_time = float(generated_time(time-prefix, reference.metadata.get("generated_time_mapping")))
                generated_end = reference.metadata.get("generated_end_seconds", reference_end)
                phase = "stand prefix" if time < prefix else ("stand suffix" if time > generated_end and reference.metadata.get("stand_suffix_frames",0) else f"motion {source_time:.2f} s")
                if post_motion_behavior == "deploy" and time >= deploy_time:
                    phase += " | deploy: paused frame 0"
                if human_renderer is not None:
                    sample = np.clip(source_time * human.fps, 0, len(joints) - 1)
                    a = int(sample); b = min(a + 1, len(joints) - 1)
                    points = joints[a] * (1 - (sample-a)) + joints[b] * (sample-a)
                    human_camera.lookat[:] = points[0]; human_camera.lookat[2] = human_center_z
                    human_renderer.update_scene(human_data, camera=human_camera)
                    scene = human_renderer.scene
                    for j, point in enumerate(points):
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.035, 0, 0]), point, np.eye(3).ravel(), np.array([.9, .4, .15, 1]))
                        scene.ngeom += 1
                        if j > 0:
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.15, .4, .8, 1]))
                            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .025, points[parents[j]], point)
                            scene.ngeom += 1
                    img = Image.fromarray(human_renderer.render()); draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, 640, 32), fill=(20, 25, 35))
                    draw.text((10, 10), f"Generated human | {phase}", fill="white")
                    images.append(np.asarray(img))
                for label, qpos in [(f"{source} reference", target[i]), ("SONIC v1.1 executed", actual[i])]:
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = qpos[:3]; camera.lookat[2] = .8
                    renderer.update_scene(data, camera=camera)
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, 640, 32), fill=(20, 25, 35))
                    drift = f" | XY error {np.linalg.norm(actual[i, :2]-target[i, :2]):.2f} m" if label.endswith("executed") else ""
                    if label.endswith("executed") and time > recorded_duration+1e-9:
                        label += f" | TERMINATED {recorded_duration:.2f}s; display held"
                    draw.text((10, 10), f"{label} | {time:.2f} s | {phase}{drift}", fill="white")
                    images.append(np.asarray(img))
                writer.append_data(np.concatenate(images, axis=1))
    finally:
        renderer.close()
        if human_renderer is not None:
            human_renderer.close()
    return dict(path=str(out), frames=len(times), fps=fps, width=1920 if human is not None else 1280,
                height=480, stand_prefix_seconds=prefix, recorded_duration_seconds=recorded_duration,
                display_duration_seconds=duration, post_motion_behavior=post_motion_behavior,
                robot_camera="Each panel follows its own root XY; executed panel displays actual XY tracking error",
                human_alignment="Hold frame zero during prefix, then sample at g(video_time - stand_prefix_seconds), freeze final human pose in suffix" if human is not None else None,
                generated_time_mapping=reference.metadata.get("generated_time_mapping"))
