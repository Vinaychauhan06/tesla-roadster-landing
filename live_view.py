"""Live animation of training: plays every attempt of car #1, in order, in the
PDF's animation style, straight from the physics the training is logging.

Runs alongside training (or on its own afterwards):

    python train_jump.py --watch               # trains and opens this viewer
    python live_view.py                        # watch a run that is already going
    python live_view.py --flip                 # the flip run (logs/flip_*.csv)
    python live_view.py --from-attempt 300     # start later

When it falls behind training it plays faster (BACKLOG on screen); it never skips an attempt.
Close the window or press Esc to stop.
"""
import argparse
import csv
import io
import os
import time

import numpy as np
import pygame

from car_jump import CarJumpEnv, Smoke, crash_burst, draw_scene, emit, post, FIGW, FIGH, DPI
from replay_attempts import crash_reason, to_frames

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--flip", action="store_true")
parser.add_argument("--log", help="frames CSV (default logs/frames_car1.csv or logs/flip_frames_car1.csv)")
parser.add_argument("--from-attempt", type=int, default=1)
parser.add_argument("--scale", type=float, default=1.25, help="window size multiplier")
args = parser.parse_args()
path = args.log or f"logs/{'flip_' if args.flip else ''}frames_car1.csv"


class Tail:
    """Reads complete attempts from the frames CSV while training keeps appending to it."""

    def __init__(self, path):
        while not os.path.exists(path):
            time.sleep(0.5)
        self.fh = open(path)
        self.header = next(csv.reader([self.fh.readline()]))
        self.partial, self.rows, self.done = "", [], {}

    def poll(self):
        chunk = self.fh.read()
        if not chunk:
            return
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()  # incomplete last line: wait for the rest
        for line in lines:
            if not line:
                continue
            values = next(csv.reader(io.StringIO(line)))
            row = {k: float(v) for k, v in zip(self.header, values)}
            row["flip_progress"] = float(values[len(self.header)]) if len(values) > len(self.header) else 0.0
            if self.rows and row["attempt"] != self.rows[-1]["attempt"]:
                self.done[int(self.rows[-1]["attempt"])] = self.rows
                self.rows = []
            self.rows.append(row)


def attempt_result(rows):
    last = rows[-1]
    landed = last["reward"] > 50          # a safe landing pays +100 or more on its last frame
    flipped = last["flip_progress"] >= (2 * np.pi - 0.35) / (2 * np.pi)
    reason = None if landed else crash_reason(last)
    return dict(landed=landed, score=last["score"], miss=abs(last["to_spot"] * 60),
                target_x=CarJumpEnv.TARGET_X, reason=reason, flipped=flipped)


pygame.init()
W, H = int(FIGW * DPI * args.scale), int(FIGH * DPI * args.scale)
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Tesla Roadster Landing - live training")
clock = pygame.time.Clock()

fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI)
fig.patch.set_facecolor("#03040a")
ax = fig.add_axes([0, 0, 1, 1])
rng = np.random.default_rng(3)
tail = Tail(path)
attempt = args.from_attempt
stats = {"landed": 0, "spot": 0, "flips": 0, "seen": 0}
running = True


def show(img):
    surf = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")
    screen.blit(pygame.transform.smoothscale(surf, (W, H)), (0, 0))
    pygame.display.flip()


while running:
    tail.poll()
    if attempt not in tail.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        pygame.time.wait(100)
        continue

    rows = tail.done.pop(attempt)
    result = attempt_result(rows)
    backlog = sum(1 for n in tail.done if n > attempt)
    stride = min(3 * (1 + backlog // 8), 15)   # play faster when behind
    frames = to_frames(rows, result["landed"], hold=30)[::stride]
    burst_i = next((i for i, f in enumerate(frames) if f["post"]), None)
    outcome = ("LANDED" if result["landed"] else "CRASHED") + (" +FLIP" if result["flipped"] else "")
    label = (f"ATTEMPT {attempt}   (LIVE TRAINING)\nTRAINING STEP {rows[-1]['train_step'] / 1e6:.2f}M\n"
             f"landed {stats['landed']}/{stats['seen']}  on spot {stats['spot']}"
             + (f"  flips {stats['flips']}" if args.flip else "")
             + (f"\nBACKLOG {backlog} - playing x{stride // 3}" if backlog > 8 else ""))

    sm, cam, wr = Smoke(), None, 0.0
    dt = CarJumpEnv.DT * stride
    for i, f in enumerate(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        emit(sm, f, rng)
        if not result["landed"] and i == burst_i:
            crash_burst(sm, f, rng)
        sm.step(dt)
        wr += f["s"] * dt / 0.45
        cam = f["x"] + 6 if cam is None else cam + (f["x"] + 6 - cam) * 0.18 * min(stride / 2, 3)
        draw_scene(ax, f, cam, wr, sm, rng, result["landed"], f["post"], target_x=result["target_x"],
                   label=label, miss=result["miss"], extras=True, reason=result["reason"])
        if f["post"] and result["flipped"]:
            ax.text(0.5, 0.72, "FULL FLIP!", transform=ax.transAxes, ha="center", color=(.35, .75, 1),
                    fontsize=12, family="monospace", weight="bold", zorder=22)
        fig.canvas.draw()
        show(post(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), rng))
        tail.poll()
        clock.tick(30)

    stats["seen"] += 1
    stats["landed"] += result["landed"]
    stats["spot"] += result["landed"] and result["miss"] < 1
    stats["flips"] += result["landed"] and result["flipped"]
    print(f"attempt {attempt:4d}  {outcome:14s} score {result['score']:7.1f}  "
          + (result["reason"] or f"{result['miss']:.1f} m from spot"),
          flush=True)
    attempt += 1

pygame.quit()
