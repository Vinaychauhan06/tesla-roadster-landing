"""Replay car #1's training attempts from logs/frames_car1.csv in the PDF's
animation style, so you can watch it learn attempt by attempt: crash into the
parked cars, clear them but miss the spot, then land on the spot.

Each attempt shows its number, the live physics, the reward and the outcome.
Also draws logs/learning_curve.png from logs/attempts.csv.

    python replay_attempts.py                  # ~60 attempts spread over training
    python replay_attempts.py --count 120
    python replay_attempts.py --attempts 1-20  # a range
"""
import argparse
import csv
import os
import tempfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from car_jump import CarJumpEnv, render_frames

DT = CarJumpEnv.DT


def load_attempts():
    by_attempt = defaultdict(list)
    with open("logs/frames_car1.csv") as fh:
        for row in csv.DictReader(fh):
            by_attempt[int(row["attempt"])].append({k: float(v) for k, v in row.items()})
    results = {}
    with open("logs/attempts.csv") as fh:
        for row in csv.DictReader(fh):
            if row["car"] == "1":
                results[int(row["car_attempt"])] = row
    return {n: rows for n, rows in by_attempt.items() if n in results}, results


def crash_reason(last):
    x, y = last["x"] * 60, last["y"] * 6   # the log holds the scaled observation
    if last["on_road"] == 1 and x > CarJumpEnv.OBSTACLES[0][0] - 2:
        return "drove into the parked cars"
    for x0, x1, h, kind, _ in CarJumpEnv.OBSTACLES:
        if x0 - 2 < x < x1 + 2 and y < h:
            return {"car": "hit a parked car", "cyber": "hit the Cybertruck", "truck": "hit the semi truck"}[kind]
    if y > 14.9 or x > CarJumpEnv.TARGET_X + 24.9:
        return "flew off - never came down"
    if x < CarJumpEnv.OBSTACLES[0][0] - 2:
        return "lifted off too early - fell back on the road"
    if x < CarJumpEnv.LAND_X:
        return "came down before the landing zone"
    return "landed too hard / tilted"


def to_frames(rows, landed, hold=45):
    frames = []
    for r in rows:
        x, y = r["x"] * 60, r["y"] * 6
        vx, vy = r["vx"] * 30, r["vy"] * 15
        a = float(np.arctan2(r["sin_tilt"], r["cos_tilt"]))
        th = 0.0 if r["on_road"] == 1 or r["fuel"] <= 0 else max(0.0, r["thrust"])
        phys = (f"x {x:5.1f} m   height {max(y, 0):4.1f} m\n"
                f"speed {vx:4.1f} / {vy:+4.1f} m/s\n"
                f"tilt {np.degrees(a):+4.0f} deg  thrust {th:.2f}\n"
                f"reward {r['reward']:+7.2f}")
        frames.append(dict(x=x, y=y, a=a, th=th, fuel=r["fuel"], s=float(np.hypot(vx, vy)),
                           post=False, score=r["score"], phys=phys))
    last = frames[-1]
    x, a, v = last["x"], last["a"], max(rows[-1]["vx"] * 30, 0.0)
    for _ in range(hold):  # the PDF's coast-to-a-stop / hold after the attempt ends
        if landed:
            v *= 0.85  # brake harder than the PDF's 0.96 so the stop stays next to the spot
            x += v * DT * 2
            a *= 0.85
            frames.append({**last, "x": x, "y": 0.0, "a": a, "th": 0.0, "s": v, "post": True})
        else:
            frames.append({**last, "y": max(last["y"], 0.0), "th": 0.0, "s": 0.0, "post": True})
    return frames


def render_one(job):
    n, rows, res, total, out = job
    landed = res["result"] != "CRASHED"
    result = dict(landed=landed, score=float(res["score"]), miss=float(res["miss_m"]),
                  target_x=CarJumpEnv.TARGET_X, reason=None if landed else crash_reason(rows[-1]))
    label = f"ATTEMPT {n} / {total}\nTRAINING STEP {int(res['train_step']) / 1e6:.2f}M"
    imgs, ms = render_frames(to_frames(rows, landed), result, stride=3, label=label, extras=True)
    import imageio.v2 as imageio
    imageio.mimwrite(out, [np.asarray(im) for im in imgs], fps=1000 / ms, codec="libx264",
                     quality=7, macro_block_size=8)
    return n, res["result"], result["reason"], float(res["score"]), float(res["miss_m"])


def pick(attempts, results, count, spec):
    ns = sorted(attempts)
    if spec:
        lo, hi = (int(v) for v in spec.split("-"))
        return [n for n in ns if lo <= n <= hi]
    chosen = set(ns[:: max(1, len(ns) // count)])
    chosen |= {ns[0], ns[-1]}
    first_landing = next((n for n in ns if results[n]["result"] != "CRASHED"), None)
    first_perfect = next((n for n in ns if results[n]["result"] == "PERFECT"), None)
    chosen |= {n for n in (first_landing, first_perfect) if n}
    return sorted(chosen)


def learning_curve(results, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(results)
    score = np.array([float(results[n]["score"]) for n in ns])
    res = [results[n]["result"] for n in ns]
    miss = np.array([float(results[n]["miss_m"]) for n in ns])
    colors = {"CRASHED": "#e5534b", "LANDED": "#e0a030", "PERFECT": "#3fb950"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, facecolor="#0d1117")
    for ax in (a1, a2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.grid(color="#21262d")
    for kind, c in colors.items():
        idx = [i for i, r in enumerate(res) if r == kind]
        a1.scatter(np.array(ns)[idx], score[idx], s=9, c=c, label=kind.lower(), zorder=3)
    k = 20
    roll = np.convolve(score, np.ones(k) / k, mode="valid")
    a1.plot(ns[k - 1:], roll, color="#c9d1d9", lw=1.5, label=f"average of last {k}")
    a1.scatter(chosen, [float(results[n]["score"]) for n in chosen], s=40, facecolors="none",
               edgecolors="#58a6ff", lw=1, label="in the replay video", zorder=4)
    a1.set_ylabel("score", color="#c9d1d9")
    a1.set_title("Car #1 learning to jump the cars and land on the spot", color="#c9d1d9")
    a1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8, loc="lower right")
    landed_idx = [i for i, r in enumerate(res) if r != "CRASHED"]
    a2.scatter(np.array(ns)[landed_idx], miss[landed_idx], s=9,
               c=[colors[res[i]] for i in landed_idx], zorder=3)
    a2.axhline(1.0, color="#3fb950", ls="--", lw=1)
    a2.text(ns[0], 1.3, "on the spot (< 1 m)", color="#3fb950", fontsize=8)
    a2.set_ylabel("landed: metres from spot", color="#c9d1d9")
    a2.set_xlabel("attempt", color="#c9d1d9")
    fig.tight_layout()
    fig.savefig("logs/learning_curve.png", dpi=110)
    print("Saved logs/learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--attempts", help="a range like 1-20")
    parser.add_argument("--out", default="learning_replay.mp4")
    args = parser.parse_args()

    attempts, results = load_attempts()
    chosen = pick(attempts, results, args.count, args.attempts)
    learning_curve(results, chosen)
    total = max(attempts)
    print(f"{len(attempts)} attempts logged; replaying {len(chosen)} of them on {os.cpu_count()} cores...")

    import imageio.v2 as imageio
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(n, attempts[n], results[n], total, f"{tmp}/{n:05d}.mp4") for n in chosen]
        with Pool() as pool:
            for n, res, reason, score, miss in pool.imap(render_one, jobs):
                print(f"  attempt {n:4d}  {res:8s} score {score:7.1f}  "
                      f"{reason or f'{miss:.1f} m from spot'}", flush=True)
        with imageio.get_writer(args.out, fps=1000 / (DT * 3 * 1000), codec="libx264",
                                quality=7, macro_block_size=8) as writer:
            for n in chosen:
                for frame in imageio.get_reader(f"{tmp}/{n:05d}.mp4"):
                    writer.append_data(frame)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
