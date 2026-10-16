"""Score the trained car, then render GIFs in the PDF's animation style.

    python make_jump_gifs.py                     # the included pretrained AI: test + car_jump_ai.gif
    python make_jump_gifs.py --test-only
    python make_jump_gifs.py --name jump_t67_s0  # your own training run (+ car_jump_learning.gif)
    python make_jump_gifs.py --name jump_t67_flip_s0 --flip
"""
import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO

from car_jump import CarJumpEnv, render_frames, rollout, save_gif

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="a training run, e.g. jump_t67_s0 (default: the included pretrained model)")
parser.add_argument("--attempts", type=int, default=20)
parser.add_argument("--test-only", action="store_true")
parser.add_argument("--flip", action="store_true", help="the model was trained with --flip")
args = parser.parse_args()


def pilot(path):
    model = PPO.load(path)
    return lambda obs: model.predict(obs, deterministic=True)[0]


def save(imgs, ms, path):
    save_gif(imgs, ms, path)
    mp4 = path[:-4] + ".mp4"
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB) and {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB)")


if args.name:
    final = pilot(f"models/{args.name}_final")
    target = float(args.name.split("_t")[1].split("_")[0])
else:
    final, target = pilot("models/pretrained_t67"), 67.0

# The test: fresh attempts, no randomness in the pilot's choices
results = [rollout(final, seed=1000 + i, hold=0, target_x=target, flip=args.flip)[1] for i in range(args.attempts)]
for i, r in enumerate(results):
    status = ("PERFECT" if r["miss"] < 1 else "LANDED ") if r["landed"] else "CRASHED"
    print(f"attempt {i + 1:2d}  {status}  score {r['score']:6.1f}  {r['miss']:5.2f} m from spot")
landed = [r for r in results if r["landed"]]
print(f"\nlanded {len(landed)}/{len(results)}, within 1 m of the spot {sum(r['miss'] < 1 for r in landed)}"
      f", mean miss {np.mean([r['miss'] for r in landed]):.2f} m, mean score {np.mean([r['score'] for r in results]):.1f}")
if args.test_only:
    raise SystemExit

# 1. The trained car in the PDF's own look: jump, fly the parked cars, land on the spot
frames, result = rollout(final, seed=1, target_x=target, flip=args.flip)
save(*render_frames(frames, result), "car_jump_ai.gif")

# 2. Learning: one attempt from snapshots taken during training, first crash to final landing
snaps = sorted(glob.glob(f"models/{args.name}_*k.zip")) if args.name else []
if len(snaps) < 3:
    print("(car_jump_learning.gif needs a training run: npm run train, then npm run demo -- --name jump_t67_s0)")
    raise SystemExit(0)
args_steps = int(snaps[-1].rsplit("_", 1)[1][:-5]) * 1000
picks = [snaps[1], snaps[len(snaps) * 5 // 9], f"models/{args.name}_final.zip"]  # ~early crash, first landing, final
all_imgs, ms = [], 40
for path in picks:
    steps = f"{args_steps // 1000}k" if path.endswith("final.zip") else path.rsplit("_", 1)[1][:-4]
    frames, result = rollout(pilot(path), seed=1, hold=50, target_x=target, flip=args.flip)
    imgs, ms = render_frames(frames, result, label=f"LEARNING: {int(steps[:-1]):,}k STEPS")
    all_imgs += imgs
    print(f"  {steps}: {'landed' if result['landed'] else 'crashed'}, score {result['score']:.1f}")
save(all_imgs, ms, "car_jump_learning.gif")
