"""Render the release GIF: the G1 walking on Holosoma's terrain under an XSoma policy, next to the headline result.

    bash tools/download_meshes.sh                                         # the G1's visual meshes, once
    python tools/rollout.py ckpt=<ckpt>.npz out=rollout.npz tile=2,4      # on a GPU machine
    python tools/render_gif.py rollout.npz                                # -> figures/xsoma.gif

The numbers on the GIF come from results/ (tools/make_figures.py), not from arguments. The camera follows the robot; the
command plan of tools/rollout.py (walk, turn, sidestep, walk backward, stand) plays out. Rendering needs OpenGL
(MUJOCO_GL=glfw by default; egl on a headless machine).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_figures import load_results, summarize  # noqa: E402

from xsoma import scene, terrain  # noqa: E402

W, H, PANEL = 960, 540, 470  # render size and the width of the robot panel on the right
BG = (255, 255, 255)  # XMimic's release palette: white, ink, the method in teal, the baselines in grays
INK, INK2 = (29, 29, 31), (72, 72, 78)
TEAL, GRAY, LIGHT = (0, 176, 190), (110, 110, 115), (174, 174, 178)


def font(size, bold=False):
    """Helvetica Neue where available (macOS), else matplotlib's DejaVu Sans."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=1 if bold else 0)
    except OSError:
        import matplotlib

        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name), size)


def model(rec):
    """The training scene (the rollout's terrain, the G1 with its visual meshes) restyled: a white sky the terrain fades into,
    light gray ground whose relief shows in a key light's shading, and soft shadows."""
    tr = terrain.generate(terrain.TerrainCfg.scaled(float(rec["scale"]), int(rec["seed"]) if "seed" in rec.files else 42))
    if not os.path.isdir(os.path.join(ROOT, "assets", "meshes")) and not os.environ.get("XSOMA_G1_XML"):
        sys.exit("the G1 meshes are missing: run bash tools/download_meshes.sh")
    spec = scene.spec(terrain=tr, visual=True)
    v = spec.visual
    v.global_.offwidth, v.global_.offheight = 2 * PANEL, 2 * H
    v.quality.shadowsize, v.quality.offsamples = 8192, 8
    v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.36] * 3, [0.10] * 3, [0.02] * 3
    v.map.fogstart, v.map.fogend, v.map.shadowclip, v.map.shadowscale = 4.0, 12.0, 3.0, 0.8
    v.rgba.fog = [1.0, 1.0, 1.0, 1.0]
    spec.stat.extent = 2.0  # MuJoCo scales the clipping planes with the scene's extent, which the terrain makes 100+ m
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1],
                     rgb2=[1, 1, 1], width=64, height=64)  # fmt: skip
    for name, d, diffuse, shadow in (("key", [-0.45, 0.6, -0.9], 0.62, True), ("fill", [0.5, -0.4, -1.0], 0.20, False)):
        light = spec.worldbody.add_light(name=name, pos=[0, 0, 6], dir=d, castshadow=shadow, diffuse=[diffuse] * 3, specular=[0.08 if shadow else 0.0] * 3)
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    for g in spec.geoms:
        if g.name.startswith("floor"):
            g.rgba = [0.80, 0.80, 0.79, 1.0]
    return spec.compile()


def robot_frames(rec, env, start, seconds, fps):
    """The robot panel at fps frames per second, interpolated from the 50 Hz recording; the camera follows the pelvis."""
    m = model(rec)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, H, PANEL)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    cam = mujoco.MjvCamera()
    qpos = rec["qpos"][:, env].astype(np.float64)
    T = len(qpos)
    k = 40  # smoothed camera target: moving average of the pelvis over +-0.8 s
    xyz = np.pad(qpos[:, :3], ((k, k), (0, 0)), mode="edge")
    look = np.stack([np.convolve(xyz[:, j], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for j in range(3)], 1)
    out = []
    for i in range(int(round(seconds * fps))):
        s = min((start + i / fps) / float(rec["dt"]), T - 1.001)
        i0, w = int(s), s - int(s)
        q = (1 - w) * qpos[i0] + w * qpos[i0 + 1]
        q0, q1 = qpos[i0, 3:7], qpos[i0 + 1, 3:7] * (1 if np.dot(qpos[i0, 3:7], qpos[i0 + 1, 3:7]) >= 0 else -1)
        q[3:7] = ((1 - w) * q0 + w * q1) / np.linalg.norm((1 - w) * q0 + w * q1)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        cam.azimuth, cam.elevation, cam.distance = 140 + 2.0 * i / fps, -16, 3.4
        cam.lookat[:] = [look[i0, 0], look[i0, 1], look[i0, 2] - 0.3]
        renderer.update_scene(d, camera=cam)
        out.append(Image.fromarray(renderer.render()))
    return out


def text_panel(summ):
    """The left panel: title, the speedup over FastSAC, and the times behind it (all from results/)."""
    img = Image.new("RGB", (W - PANEL, H), BG)
    d = ImageDraw.Draw(img)
    x = 40
    t_x, t_f = np.mean(summ["xsoma"]["held"]), np.mean(summ["fast_sac"]["held"])
    d.text((x, 40), "XSoma", font=font(48, bold=True), fill=INK)
    d.text((x - 6, 96), f"{t_f / t_x:.1f}×", font=font(150, bold=True), fill=TEAL)
    d.text((x, 266), "faster than FastSAC to", font=font(32), fill=INK)
    d.text((x, 305), "Holosoma's tracking bar", font=font(32), fill=INK)
    num = font(40, bold=True)
    rows = [(f"{t_x:.1f} min", TEAL, "XSoma"), (f"{t_f:.1f} min", GRAY, "FastSAC")]
    if "ppo" in summ:
        ppo = summ["ppo"]["held"]
        rows.append((f"{np.mean(ppo):.1f} min" if ppo else f"> {5 * round(summ['ppo']['end'] / 5):.0f} min", LIGHT, "PPO"))  # the run length
    off = int(max(d.textlength(r[0], font=num) for r in rows)) + 16
    for j, (value, color, label) in enumerate(rows):
        y = 372 + 50 * j
        d.text((x, y), value, font=num, fill=color)
        d.text((x + off, y + 12), label + ", 1 H100", font=font(26), fill=INK2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout", help="tools/rollout.py output")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "xsoma.gif"))
    ap.add_argument("--env", type=int, default=0, help="which of the recorded robots")
    ap.add_argument("--start", type=float, default=0.5, help="start in the recording (s)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--fade", type=int, default=6, help="frames of fade at the loop boundary")
    ap.add_argument("--scale", type=float, default=0.75, help="output size relative to 960 x 540 (0.75: 720 x 405 for the README)")
    ap.add_argument("--colors", type=int, default=96)
    args = ap.parse_args()
    rec = np.load(args.rollout)
    left = text_panel(summarize(load_results()))
    robots = robot_frames(rec, args.env, args.start, args.seconds, args.fps)
    frames = []
    for i, rob in enumerate(robots):
        a = min(1.0, (i + 1) / args.fade, (len(robots) - i) / args.fade)  # fade the robot in and out at the loop boundary
        rob = Image.blend(Image.new("RGB", rob.size, BG), rob, a) if a < 1 else rob
        f = Image.new("RGB", (W, H), BG)
        f.paste(left, (0, 0))
        f.paste(rob, (W - PANEL, 0))
        frames.append(f)
    if args.scale != 1.0:
        frames = [f.resize((round(W * args.scale), round(H * args.scale)), Image.Resampling.LANCZOS) for f in frames]
    w, h = frames[0].size
    # one palette for all frames (no flicker), from a montage of samples
    sample = Image.new("RGB", (w, h * 4))
    for j, f in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        sample.paste(f, (0, h * j))
    pal = sample.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0, optimize=True)
    print(f"{len(frames)} frames {w}x{h} at {args.fps} fps -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
