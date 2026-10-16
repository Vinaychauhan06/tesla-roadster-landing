"""Teach the car to jump the parked cars and land on the spot, with PPO.

16 cars practise in parallel. Car #1 is shown live: every attempt, with its
observation (the 13 numbers it sees), action, and reward every few frames,
then the attempt's result. Everything is also written to logs/:

  logs/attempts.csv     every attempt of all 16 cars (score, result, miss, frames)
  logs/frames_car1.csv  every frame of car #1 (13 observations, 3 actions, reward, score)

    python train_jump.py
    python train_jump.py --watch                 # also animate every attempt live
    python train_jump.py --frame-every 10        # show more frames per attempt
    python train_jump.py --steps 3000000 --target 75
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from car_jump import CarJumpEnv

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=2_000_000)
parser.add_argument("--target", type=float, default=CarJumpEnv.TARGET_X)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--snapshots", type=int, default=8)
parser.add_argument("--flip", action="store_true", help="teach a full mid-air flip before landing")
parser.add_argument("--watch", action="store_true", help="open the live animation of every attempt")
parser.add_argument("--frame-every", type=int, default=25, help="print car #1's observation every N frames (0 = off)")
args = parser.parse_args()

name = f"jump_t{args.target:g}{'_flip' if args.flip else ''}_s{args.seed}"
LOG = f"logs/{'flip_' if args.flip else ''}"
OBS_NAMES = ["x", "y", "vx", "vy", "sin_tilt", "cos_tilt", "spin", "on_road", "fuel",
             "to_launch", "to_cars", "to_land_zone", "to_spot"]
ACT_NAMES = ["throttle", "thrust", "torque"]


def make():
    return Monitor(CarJumpEnv(args.target, args.flip), info_keywords=("landed", "miss", "flipped"))


def result_of(info):
    flip = "+FLIP" if args.flip and info.get("flipped") else ""
    if info["landed"]:
        return ("PERFECT" if info["miss"] < 1.0 else "LANDED") + flip
    return "CRASHED" + flip


class Scoreboard(BaseCallback):
    def __init__(self):
        super().__init__()
        self.attempt = 0                # car #1's attempts
        self.attempts_all = 0           # every car's attempts
        self.recent = []
        self.next_snap = 0
        self.frame = 0                  # frame within car #1's current attempt
        self.score = 0.0

        os.makedirs("logs", exist_ok=True)
        self.attempts_file = open(LOG + "attempts.csv", "w", newline="")
        self.attempts_csv = csv.writer(self.attempts_file)
        self.attempts_csv.writerow(["attempt_all", "car", "car_attempt", "train_step", "frames",
                                    "score", "result", "miss_m"])
        self.frames_file = open(LOG + "frames_car1.csv", "w", newline="")
        self.frames_csv = csv.writer(self.frames_file)
        self.frames_csv.writerow(["attempt", "frame", "train_step", *ACT_NAMES, "reward", "score", *OBS_NAMES])
        self.car_attempts = [0] * 16

    def _header(self):
        print(f"\n=== car #1  attempt {self.attempt + 1}  [{self.num_timesteps / 1e6:.2f}M steps] ===", flush=True)
        if args.frame_every:
            print("  frame |   x     y    vx    vy  tilt  spin road fuel spot | thrtl thrst torq | reward   score")

    def _on_training_start(self):
        self._header()

    def _on_step(self):
        if self.num_timesteps >= self.next_snap:
            self.model.save(f"models/{name}_{self.num_timesteps // 1000:05d}k")
            self.next_snap += args.steps // args.snapshots

        # ---- car #1, frame by frame
        info0 = self.locals["infos"][0]
        done0 = self.locals["dones"][0]
        obs0 = info0["terminal_observation"] if done0 else self.locals["new_obs"][0]
        act0 = self.locals["clipped_actions"][0]
        rew0 = float(self.locals["rewards"][0])
        self.frame += 1
        self.score += rew0
        self.frames_csv.writerow([self.attempt + 1, self.frame, self.num_timesteps,
                                  *np.round(act0, 3), round(rew0, 3), round(self.score, 2),
                                  *np.round(obs0, 4)])
        if args.frame_every and (self.frame % args.frame_every == 1 or done0):
            x, y, vx, vy = obs0[0] * 60, obs0[1] * 6, obs0[2] * 30, obs0[3] * 15
            tilt = np.degrees(np.arctan2(obs0[4], obs0[5]))
            print(f"  {self.frame:5d} | {x:5.1f} {y:5.2f} {vx:5.1f} {vy:5.1f} {tilt:5.0f} {obs0[6] * 5:5.1f}"
                  f"  {int(obs0[7])}  {obs0[8]:.2f} {obs0[12] * 60:5.1f} |"
                  f" {act0[0]:5.2f} {max(act0[1], 0):5.2f} {act0[2]:5.2f} | {rew0:6.2f} {self.score:7.1f}",
                  flush=True)

        # ---- every car: attempt results
        for i, info in enumerate(self.locals["infos"]):
            ep = info.get("episode")
            if ep is None:
                continue
            self.attempts_all += 1
            self.car_attempts[i] += 1
            self.recent = (self.recent + [(ep["r"], info["landed"], info["miss"], info["flipped"])])[-100:]
            self.attempts_csv.writerow([self.attempts_all, i + 1, self.car_attempts[i], self.num_timesteps,
                                        ep["l"], round(ep["r"], 2), result_of(info), round(info["miss"], 2)])
            if i != 0:
                continue
            self.attempt += 1
            where = f"{info['miss']:.1f} m from spot" if info["landed"] else f"ended at x={obs0[0] * 60:.1f} m"
            landed = sum(r[1] for r in self.recent)
            on_spot = sum(r[1] and r[2] < 1.0 for r in self.recent)
            flips = sum(r[1] and r[3] for r in self.recent)
            print(f"  >>> {result_of(info)}  score {ep['r']:.1f}  frames {ep['l']}  {where}"
                  f"   | last 100 attempts (all cars): avg {np.mean([r[0] for r in self.recent]):.1f}, "
                  f"landed {landed}, on spot {on_spot}" + (f", flip+landed {flips}" if args.flip else ""),
                  flush=True)
            self.frame, self.score = 0, 0.0
            self.frames_file.flush()      # let live_view.py see the finished attempt straight away
            self.attempts_file.flush()
            self._header()
        return True

    def _on_training_end(self):
        self.attempts_file.close()
        self.frames_file.close()


env = DummyVecEnv([make for _ in range(16)])
env.seed(args.seed)
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.0,
            seed=args.seed, verbose=0)

if args.watch:
    viewer = [sys.executable, "live_view.py", "--log", LOG + "frames_car1.csv"] + (["--flip"] if args.flip else [])
    subprocess.Popen(viewer)

start = time.time()
model.learn(total_timesteps=args.steps, callback=Scoreboard())
model.save(f"models/{name}_final")
print(f"\nTraining took {(time.time() - start) / 60:.1f} min -> models/{name}_final.zip")
print(f"Logs: {LOG}attempts.csv (every attempt, all cars), {LOG}frames_car1.csv (every frame of car #1)")
