"""A ~45-second square (1080x1080) social-media video of a training run, made from
real logged attempts: crashes (with extra crash effects), first landing, getting
closer, and the best landing in slow motion, then an end card with test results.

Scenes are picked automatically from logs/ (so run training first):

    npm run train                      # or: python train_jump.py
    npm run video                      # -> roadster_video.mp4
    npm run video -- --out ~/Desktop/post.mp4
    npm run video -- --reel                   # 1080x1920 for Instagram Reels / TikTok / Shorts
    npm run video -- --reel --cta "DM me ROADSTER for the GitHub link"
    npm run video -- --reel --resume --badge "AI PROJECT FOR YOUR RESUME" --cta 'Comment "LINK" and I'll DM you the GitHub'
    npm run video -- --no-music

It comes with an original synthesised soundtrack (soundtrack.py): music plus jet hiss,
crash booms, touchdown thuds and a chime on the perfect landing, timed to the video.

The crash tumble, fire, flash and shake are video effects only; in the simulation an
attempt simply ends at the moment of impact.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from car_jump import CarJumpEnv, DPI, FIGH, FIGW, H_M, Y_BOTTOM, Smoke, crash_burst, draw_scene, emit, post, rollout
from replay_attempts import crash_reason, load_attempts, to_frames
from soundtrack import make_soundtrack

FPS = 30
SIZE = 1080  # width
# Where things go. A reel keeps text out of the top ~220 px and bottom ~420 px that the app's buttons cover.
LAYOUTS = {
    "square": dict(H=1080, head_y=95, sub_y=170, head=58, sub=40, scene_y=250, caption_gap=45,
                   card=[300, 385, 520, 590, 700, 830, 885]),
    "reel": dict(H=1920, head_y=420, sub_y=500, head=66, sub=44, scene_y=560, scene_h=940, caption_gap=None,
                 card=[520, 610, 760, 835, 960, 1140, 1205]),
}
L = LAYOUTS[os.environ.get("ROADSTER_LAYOUT", "square")]  # read in worker processes too
REPO = "github.com/Vinaychauhan06/tesla-roadster-landing"
FONTS = {  # first one found is used (macOS, Windows, Linux)
    "bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono": ["/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/consola.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}
BOLD, REG, MONO = "bold", "regular", "mono"
BG = (5, 6, 11)
GREEN, RED, AMBER, GREY = (63, 185, 80), (229, 83, 75), (224, 160, 48), (150, 156, 170)

# A scene is (attempt, seconds, headline, subline, colour, mode, total attempts)
#   mode "crash": slow motion into the impact, then the wreck tumbles with fire and smoke
#   mode "slowmo": slow motion touchdown and a long hold on the landing


def pick_scenes():
    """Choose the story from the logged run: crashes, flying off, first landing, closer, best."""
    attempts, results = load_attempts()
    ns = sorted(attempts)
    total = ns[-1]
    res = {n: results[n]["result"] for n in ns}
    miss = {n: float(results[n]["miss_m"]) for n in ns}
    reason = {n: crash_reason(attempts[n][-1]) for n in ns if res[n] == "CRASHED"}
    first_landing = next((n for n in ns if res[n] != "CRASHED"), None)
    if first_landing is None:
        sys.exit("No landing in this run yet - train longer (npm run train -- --steps 3000000).")

    def first(pred, after=0):
        return next((n for n in ns if n > after and pred(n)), None)

    crashes = []
    for want in ("lifted off", "drove into", "hit a parked car", "hit the semi", "hit the Cybertruck"):
        n = first(lambda n: n in reason and reason[n].startswith(want) and n not in crashes)
        if n:
            crashes.append(n)
    crashes = sorted(crashes[:4])
    before = sum(1 for n in ns if n < first_landing)
    flew = first(lambda n: n in reason and reason[n].startswith("flew off"))
    landed = [n for n in ns if res[n] != "CRASHED"]
    closer = [first(lambda n: res[n] != "CRASHED" and lo <= miss[n] < hi, first_landing)
              for lo, hi in ((7, 15), (1.2, 4))]
    late = [n for n in landed if n > ns[int(len(ns) * 0.7)]] or landed
    best = min(late, key=lambda n: miss[n])

    lines = [("This AI crashed {before} times".format(before=before), "before it learned to fly."),
             ("Nobody told it how to fly.", None),
             ("It only gets a score:", "-100 for every crash."),
             (None, "Every crash teaches it something.")]
    # crashes and the fly-off in the order they happened, so the attempt counter only goes up
    early = sorted(crashes + ([flew] if flew else []))
    texts = iter(lines)
    scenes = []
    for n in early:
        if n == flew:
            scenes.append((n, 3.4, f"Attempt {n}: it learned to fly...", "...but never came down.",
                           AMBER, "normal", total))
            continue
        head, subl = next(texts)
        scenes.append((n, 3.4, head or f"Attempt {n}: {reason[n]}.", subl or f"Attempt {n}: {reason[n]}.",
                       RED, "crash", total))
    scenes.append((first_landing, 4.0, f"Attempt {first_landing}: first landing!",
                   f"{miss[first_landing]:.1f} m from the target.", AMBER, "normal", total))
    for text, n in zip(("Getting closer...", "Closer..."), closer):
        if n and n < best:
            scenes.append((n, 3.2, text, f"{miss[n]:.1f} m from the target.", AMBER, "normal", total))
    scenes.append((best, 9.5, f"Attempt {best}: on the spot.", f"{miss[best]:.2f} m from the target.",
                   GREEN, "slowmo", total))
    return scenes, first(lambda n: res[n] == "PERFECT")


IMPACT = 20      # sim frames before the crash shown in slow motion
WRECK = 85       # sim frames of aftermath


def font(kind, size):
    for path in FONTS[kind]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def fitted(kind, size, text, width=SIZE - 70):
    """Largest font up to `size` that fits the text in the frame."""
    f = font(kind, size)
    while size > 20 and f.getlength(text) > width:
        size -= 2
        f = font(kind, size)
    return f


def frame_indices(n_sim, n_post, seconds, mode):
    real = 50 / FPS
    if mode == "normal":
        return np.linspace(0, n_sim + n_post - 1, int(seconds * FPS))
    if mode == "slowmo":
        touch = n_sim - 40
        return np.concatenate([np.arange(0, touch, real), np.arange(touch, n_sim, 0.35),
                               np.arange(n_sim, n_sim + n_post, real)])
    # crash: fast-forward the approach, slow motion into the impact, wreck at ~60% speed
    impact = max(n_sim - IMPACT, 0)
    slow = np.arange(impact, n_sim, 0.4)
    wreck = np.arange(n_sim, n_sim + n_post, 1.0)
    budget = max(int(seconds * FPS) - len(slow) - len(wreck), 8)
    approach = np.linspace(0, impact, budget, endpoint=False) if impact > 0 else np.array([])
    return np.concatenate([approach, slow, wreck])


def wreck_frames(rows, last, rng):
    """After the impact: the car bounces off, tumbles and falls to the road."""
    x, y, a = last["x"], max(last["y"], 0.0), last["a"]
    vx = rows[-1]["vx"] * 30 * -0.25            # knocked back
    vy = max(rows[-1]["vy"] * 15 * 0.3, 0.0) + 3.0  # popped up by the hit
    spin = rng.choice([-1, 1]) * rng.uniform(5, 8)
    out = []
    for _ in range(WRECK):
        vy -= CarJumpEnv.G * CarJumpEnv.DT
        x += vx * CarJumpEnv.DT
        y += vy * CarJumpEnv.DT
        a += spin * CarJumpEnv.DT
        if y <= 0:
            y = 0.0
            vy = -vy * 0.3 if abs(vy) > 2 else 0.0
            vx *= 0.6
            spin *= 0.55
        out.append({**last, "x": x, "y": y, "a": a, "th": 0.0, "s": 0.0, "post": True, "wreck": True})
    return out


def wreck_fx(sm, f, rng):
    """Fire and black smoke pouring off the wreck."""
    for _ in range(2):
        sm.add(f["x"] + rng.normal(0, .5), f["y"] + 0.9, rng.normal(0, .6), rng.uniform(1, 3),
               rng.uniform(.3, .6), .35, 1.8, (1.0, rng.uniform(.4, .65), .12), .55, buoy=1.2, drag=1.2)
    sm.add(f["x"] + rng.normal(0, .4), f["y"] + 1.1, rng.normal(0, .4), rng.uniform(.8, 2),
           rng.uniform(1.5, 2.5), .5, 2.6, (.16, .16, .18), .45, buoy=.9, drag=.8)


def render_scene(scene):
    n, seconds, head, sub, colour, mode, total = scene
    attempts, results = load_attempts()
    rows, res = attempts[n], results[n]
    landed = res["result"] != "CRASHED"
    rng = np.random.default_rng(n)
    if mode == "crash":
        frames = to_frames(rows, landed, hold=0)
        frames += wreck_frames(rows, frames[-1], rng)
    else:
        frames = to_frames(rows, landed, hold=110 if mode == "slowmo" else 45)
    for f in frames:
        f.pop("phys", None)
    n_post = sum(f["post"] for f in frames)
    idx = frame_indices(len(frames) - n_post, n_post, seconds, mode)
    result = dict(landed=landed, miss=float(res["miss_m"]), target_x=CarJumpEnv.TARGET_X,
                  reason=None if landed else crash_reason(rows[-1]))

    fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI)
    fig.patch.set_facecolor("#03040a")
    ax = fig.add_axes([0, 0, 1, 1])
    sm, cam, wr, prev, impact_at, out, flashes = Smoke(), None, 0.0, 0.0, None, [], []
    touchdown_at, thrust = None, []
    for t in idx:
        f = frames[int(t)]
        dt = max((t - prev) * CarJumpEnv.DT, CarJumpEnv.DT * 0.3)
        prev = t
        emit(sm, f, rng)
        if not landed and f["post"] and impact_at is None:
            impact_at = len(out)
            for _ in range(2 if mode == "crash" else 1):
                crash_burst(sm, f, rng)
        if landed and f["post"] and touchdown_at is None:
            touchdown_at = len(out)
        if f.get("wreck"):
            wreck_fx(sm, f, rng)
        sm.step(dt)
        wr += f["s"] * dt / 0.45
        follow = min(0.18 * dt / 0.02, 0.6)
        cam = f["x"] + 6 if cam is None else cam + (f["x"] + 6 - cam) * follow
        since = None if impact_at is None else len(out) - impact_at
        shake = 0.0 if since is None else 0.7 * max(0.0, 1 - since / 14)
        draw_scene(ax, f, cam + rng.normal(0, shake), wr, sm, rng, landed, f["post"],
                   target_x=result["target_x"], miss=result["miss"], extras=True, reason=result["reason"])
        if shake:
            dy = rng.normal(0, shake * 0.6)
            ax.set_ylim(Y_BOTTOM + dy, Y_BOTTOM + H_M + dy)
        fig.canvas.draw()
        out.append(post(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), rng))
        flashes.append(0.0 if since is None else 0.55 * max(0.0, 1 - since / 6))
        thrust.append(f["th"])
    plt.close(fig)
    events = dict(frames=len(out), thrust=thrust, impact=impact_at if mode == "crash" else None,
                  touchdown=touchdown_at, perfect=landed and mode == "slowmo")
    return [compose(img, n, total, head, sub, colour, i / len(out), flashes[i]) for i, img in enumerate(out)], events


def compose(scene_img, attempt, total, head, sub, colour, t, flash=0.0):
    canvas = Image.new("RGB", (SIZE, L["H"]), BG)
    reel = "scene_h" in L
    if reel:
        # fill the phone screen: scale the scene up and crop a window around the car (it sits ~40% from the left)
        h = L["scene_h"]
        big = Image.fromarray(scene_img).resize((int(scene_img.shape[1] * h / scene_img.shape[0]), h), Image.LANCZOS)
        x0 = int(np.clip(0.44 * big.width - SIZE / 2, 0, big.width - SIZE))
        scene = big.crop((x0, 0, x0 + SIZE, h))
    else:
        h = int(SIZE * scene_img.shape[0] / scene_img.shape[1])
        scene = Image.fromarray(scene_img).resize((SIZE, h), Image.LANCZOS)
    y0 = L["scene_y"]
    if flash:  # red flash on impact
        scene = Image.blend(scene, Image.new("RGB", scene.size, (255, 70, 40)), flash)
    canvas.paste(scene, (0, y0))
    d = ImageDraw.Draw(canvas)
    badge = os.environ.get("ROADSTER_BADGE")
    if badge:  # e.g. "AI PROJECT FOR YOUR RESUME", a pill over the road
        bf = font(BOLD, 34)
        bw = bf.getlength(badge) + 56
        by = y0 + h - 78
        d.rounded_rectangle(((SIZE - bw) / 2, by, (SIZE + bw) / 2, by + 56), 28, fill=(63, 185, 80))
        d.text((SIZE // 2, by + 28), badge, font=bf, fill=(6, 22, 10), anchor="mm")
    fade = min(1.0, t * 6)  # headline fades in over the first part of the scene
    c = tuple(int(v * fade + b * (1 - fade)) for v, b in zip((255, 255, 255), BG))
    d.text((SIZE // 2, L["head_y"]), head, font=fitted(BOLD, L["head"], head), fill=c, anchor="mm")
    sc = tuple(int(v * fade + b * (1 - fade)) for v, b in zip(colour, BG))
    d.text((SIZE // 2, L["sub_y"]), sub, font=fitted(REG, L["sub"], sub), fill=sc, anchor="mm")

    # attempt counter and a progress bar through training (above the headline on a reel,
    # because the app's caption covers the bottom of the screen)
    yb = 250 if reel else y0 + h + 40
    d.text((60, yb), f"ATTEMPT {attempt}", font=font(MONO, 40), fill=(235, 238, 245))
    d.text((SIZE - 60, yb + 6), f"of {total}", font=font(MONO, 30), fill=GREY, anchor="ra")
    bar_y = yb + 66
    d.rounded_rectangle((60, bar_y, SIZE - 60, bar_y + 14), 7, fill=(32, 36, 46))
    d.rounded_rectangle((60, bar_y, 60 + (SIZE - 120) * attempt / total, bar_y + 14), 7, fill=colour)
    cap_y = y0 + h + 50 if reel else L["H"] - L["caption_gap"]
    d.text((SIZE // 2, cap_y), "Reinforcement learning  ·  Gymnasium + PPO",
           font=font(REG, 28), fill=GREY, anchor="mm")
    return np.asarray(canvas)


def test_model(path, attempts=20):
    """The end card's numbers: fresh attempts by the run's final model, no randomness."""
    from stable_baselines3 import PPO
    model = PPO.load(path)
    res = [rollout(lambda o: model.predict(o, deterministic=True)[0], seed=1000 + i, hold=0)[1]
           for i in range(attempts)]
    on_spot = sum(r["landed"] and r["miss"] < 1 for r in res)
    landed = [r["miss"] for r in res if r["landed"]]
    return on_spot, attempts, (np.mean(landed) if landed else None)


def end_card(last_frame, stats, first_perfect, cta=None, resume=False, seconds=None):
    base = Image.fromarray(last_frame).filter(ImageFilter.GaussianBlur(14))
    base = Image.blend(base, Image.new("RGB", base.size, BG), 0.6)
    white, soft, blue = (255, 255, 255), (220, 224, 232), (120, 180, 255)
    lines = []  # (font, text, colour, space above, pulses)
    if stats:
        on_spot, n, mean_miss = stats
        lines += [(font(BOLD, 96), f"{on_spot} / {n}", white, 0, False),
                  (font(REG, 44), "landings on the spot", soft, 70, False)]
        if mean_miss is not None:
            lines += [(font(BOLD, 64), f"{mean_miss:.2f} m", GREEN, 120, False),
                      (font(REG, 40), "average distance from the target", soft, 62, False)]
    if first_perfect:
        lines += [(font(REG, 40), f"Learned it in {first_perfect} attempts", soft, 95, False)]
    if resume:
        lines += [(fitted(BOLD, 62, "ADD THIS TO YOUR RESUME"), "ADD THIS TO YOUR RESUME", GREEN, 150, False),
                  (font(REG, 34), "Python  ·  Gymnasium  ·  Reinforcement learning", GREY, 62, False)]
    if cta:
        lines += [(fitted(BOLD, 54, cta), cta, blue, 140, True)]
        if not resume:
            lines += [(font(REG, 34), "Python  ·  Gymnasium  ·  Reinforcement learning", GREY, 62, False)]
    else:
        lines += [(font(MONO, 30), REPO, blue, 140, False),
                  (font(REG, 30), "Open source: link in the comments", GREY, 55, False)]
    height = sum(gap for *_, gap, _ in lines)
    y = (L["H"] - height) / 2 - (60 if "scene_h" in L else 0)   # a little high on a reel, clear of the caption
    placed = []
    for fnt, text, col, gap, pulse in lines:
        y += gap
        placed.append((fnt, text, col, y, pulse))

    seconds = seconds or (7.0 if cta else 5.5)
    out = []
    for i in range(int(seconds * FPS)):
        im = base.copy()
        d = ImageDraw.Draw(im)
        for k, (fnt, text, col, y, pulse) in enumerate(placed):
            a = np.clip((i - k * 4) / 10, 0, 1)  # each line fades in after the previous one
            c = tuple(int(v * a + b * (1 - a)) for v, b in zip(col, BG))
            if pulse and a == 1:  # gently pulsing call to action, boxed
                glow = 0.5 + 0.5 * np.sin(i / FPS * 2 * np.pi * 0.8)
                w = fnt.getlength(text) + 60
                d.rounded_rectangle(((SIZE - w) / 2, y - 48, (SIZE + w) / 2, y + 48), 24,
                                    outline=tuple(int(v * (0.5 + 0.5 * glow)) for v in col), width=4)
            d.text((SIZE // 2, y), text, font=fnt, fill=c, anchor="mm")
        out.append(np.asarray(im))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="roadster_video.mp4")
    parser.add_argument("--name", default="jump_t67_s0", help="the training run whose final model is tested")
    parser.add_argument("--reel", action="store_true", help="vertical 1080x1920 (Reels, TikTok, Shorts)")
    parser.add_argument("--cta", help='end-card call to action, e.g. "DM me ROADSTER for the GitHub link"')
    parser.add_argument("--resume", action="store_true", help='end card says "ADD THIS TO YOUR RESUME"')
    parser.add_argument("--badge", help='a pill shown over every scene, e.g. "AI PROJECT FOR YOUR RESUME"')
    parser.add_argument("--no-music", action="store_true", help="silent video, no soundtrack")
    args = parser.parse_args()
    if args.badge:
        os.environ["ROADSTER_BADGE"] = args.badge
    os.environ["ROADSTER_LAYOUT"] = "reel" if args.reel else "square"   # picked up by the workers
    L = LAYOUTS[os.environ["ROADSTER_LAYOUT"]]
    OUT = os.path.expanduser(args.out)
    if not os.path.exists("logs/frames_car1.csv"):
        sys.exit("No training logs yet. Run training first:  npm run train")

    scenes, first_perfect = pick_scenes()
    for sc in scenes:
        print(f"  scene: attempt {sc[0]:4d}  {sc[1]} / {sc[2]}")
    model = f"models/{args.name}_final.zip"
    stats = test_model(model) if os.path.exists(model) else None
    print(f"Rendering {len(scenes)} scenes on {os.cpu_count()} cores (a few minutes) ...")
    import tempfile
    tmp = tempfile.mkdtemp()
    silent = OUT if args.no_music else os.path.join(tmp, "video.mp4")
    writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=None, macro_block_size=8,
                                output_params=["-crf", "20", "-preset", "slow", "-movflags", "+faststart"])
    timeline, frame_no, last = [], 0, None
    with Pool() as pool:
        for frames, ev in pool.imap(render_scene, scenes):   # written as they arrive, in order
            for f in frames:
                writer.append_data(f)
            timeline.append(dict(start=frame_no / FPS, thrust=ev["thrust"], perfect=ev["perfect"],
                                 impact=None if ev["impact"] is None else (frame_no + ev["impact"]) / FPS,
                                 touchdown=None if ev["touchdown"] is None else (frame_no + ev["touchdown"]) / FPS))
            frame_no += len(frames)
            last = frames[-1]
            del frames
    card_start = frame_no / FPS
    card = end_card(last, stats, first_perfect, args.cta, args.resume)
    for f in card:
        writer.append_data(f)
    frame_no += len(card)
    writer.close()
    duration = frame_no / FPS

    if not args.no_music:
        wav = os.path.join(tmp, "music.wav")
        make_soundtrack(wav, duration, FPS, timeline, card_start)
        import subprocess
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                        OUT], check=True)
    print(f"Saved {OUT}: {duration:.1f} s, {os.path.getsize(OUT) / 1e6:.1f} MB"
          + ("" if args.no_music else ", with soundtrack"))
