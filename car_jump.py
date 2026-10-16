"""Car jet-pack jump over cars, from "Car jet-pack jump over cars: GIF code".

The environment, scripted pilot and GIF renderer are the PDF's code. Changes
for reinforcement learning are marked "RL:":
  * a reward (the original always returned 0.0, so there was nothing to learn from)
  * a fixed landing spot TARGET_X; touching down closer to it pays more
  * driving into a parked car on the road now counts as a crash
  * the observation gains a 13th number: distance to the landing spot
The renderer is unchanged apart from a spot marker and a SCORE readout.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from PIL import Image as PILImage, ImageFilter


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------- ENVIRONMENT (no ramp: jets lift the car off) ----------------
class CarJumpEnv(gym.Env):
    G, DT = 9.8, 0.02
    MAX_SPEED = 22.0
    THRUST_ACC = 16.0
    FUEL_BURN = 0.15
    LAUNCH_X = 14.0          # jets fire here
    DESCEND_X = 60.5         # start coming down after the last vehicle
    LAND_X = 63.0
    TARGET_X = 67.0          # RL: the exact spot to land on
    OBSTACLES = [  # (x_start, x_end, height, kind, color)
        (30, 34, 1.4, "car", "#e8e8e8"),
        (35, 41, 1.95, "cyber", "#b9bec6"),   # Cybertruck
        (42, 48, 3.2, "truck", "#f2f2f2"),    # semi truck
        (49, 53, 1.4, "car", "#3a6ea5"),
        (54, 58, 1.4, "car", "#777777"),
    ]

    FLIP_BONUS = 150.0       # RL flip mode: paid only for a full turn that ends in a safe landing
    FLIP_PROGRESS = 15.0     # RL flip mode: shaping per radian turned, capped at one full turn

    def __init__(self, target_x=None, flip=False):
        super().__init__()
        if target_x is not None:
            self.TARGET_X = float(target_x)
        self.flip = flip
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14 if flip else 13,), dtype=np.float32)

    @property
    def flipped(self):
        # the angle is never wrapped, so it counts total turn since lift-off
        return abs(self.angle) >= 2 * np.pi - 0.35

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.y = self.s = 0.0
        self.vx = self.vy = 0.0
        self.angle = self.ang_vel = 0.0
        self.fuel = 1.0
        self.last_thrust = 0.0
        self.airborne = False
        self.steps = 0
        self.score = 0.0
        self.prev_potential = self._potential()
        return self._obs(), {}

    def _obs(self):
        return np.array([
            self.x / 60, self.y / 6, self.vx / 30, self.vy / 15,
            np.sin(self.angle), np.cos(self.angle), self.ang_vel / 5,
            float(not self.airborne), self.fuel,
            (self.LAUNCH_X - self.x) / 60,
            (30.0 - self.x) / 60,
            (self.LAND_X - self.x) / 60,
            (self.TARGET_X - self.x) / 60,   # RL: where to land
        ] + ([min(abs(self.angle) / (2 * np.pi), 1.0)] if self.flip else []),  # RL: how much of the flip is done
            dtype=np.float32)

    def _potential(self):
        # RL: "how good is this position" -- closer to the spot and level is better.
        # Paying the change between frames (like Lunar Lander) means hovering earns nothing.
        distance = np.hypot(self.TARGET_X - self.x, self.y)   # straight line to the spot on the ground
        speed = np.hypot(self.vx, self.vy)
        if not self.flip:
            return -1.5 * distance - 0.5 * speed - 10.0 * abs(wrap(self.angle))
        # Flip mode: no tilt penalty until the full turn is done, and credit for turning (capped,
        # and it's a potential, so spinning back un-earns it -- endless spinning pays nothing)
        turn = self.FLIP_PROGRESS * min(abs(self.angle), 2 * np.pi)
        tilt = 10.0 * abs(wrap(self.angle)) if self.flipped else 0.0
        return -1.5 * distance - 0.5 * speed + turn - tilt

    def step(self, action):
        throttle, thrust, torque = np.clip(action, -1, 1)
        thrust = max(0.0, float(thrust))
        self.steps += 1
        self.last_thrust = 0.0
        terminated, info = False, {}
        crashed = False
        if not self.airborne:
            self.s += (throttle * 8.0 - 0.02 * self.s) * self.DT
            self.s = float(np.clip(self.s, 0.0, self.MAX_SPEED))
            self.x += self.s * self.DT
            self.y, self.angle = 0.0, 0.0
            self.vx, self.vy = self.s, 0.0
            if thrust > 0.7 and self.fuel > 0:          # jets strong enough to lift off
                self.airborne = True
                self.vy = 1.0
                self.ang_vel = float(self.np_random.uniform(-1.0, 1.0))
            elif self.x > self.OBSTACLES[0][0] - 2.0:   # RL: drove into the parked cars
                crashed = True
        else:
            if self.fuel <= 0:
                thrust = 0.0
            self.fuel = max(0.0, self.fuel - self.FUEL_BURN * thrust * self.DT)
            self.last_thrust = thrust
            ax_ = -np.sin(self.angle) * self.THRUST_ACC * thrust
            ay_ = np.cos(self.angle) * self.THRUST_ACC * thrust - self.G
            self.ang_vel += torque * 6.0 * self.DT
            self.angle += self.ang_vel * self.DT
            self.vx += ax_ * self.DT
            self.vy += ay_ * self.DT
            self.x += self.vx * self.DT
            self.y += self.vy * self.DT
            for o in self.OBSTACLES:
                if o[0] - 2.0 < self.x < o[1] + 2.0 and self.y < o[2]:
                    crashed = True
            touch = self.y <= 0 and self.vy < 0
            if touch and self.x < self.LAND_X:
                crashed = True
            if not crashed and touch:
                terminated = True
                info["landed"] = bool(abs(wrap(self.angle)) < 0.35 and self.vy > -6)
            # RL: flew off the scene
            if self.y > 15 or self.x > self.TARGET_X + 25:
                crashed = True
        if crashed:
            terminated = True
            info["landed"] = False

        # RL: the reward
        potential = self._potential()
        reward = potential - self.prev_potential
        self.prev_potential = potential
        reward -= 0.05                       # small fee per frame: waiting costs something
        reward -= 0.3 * self.last_thrust     # jet fuel
        if terminated:
            info["miss"] = abs(self.x - self.TARGET_X)
            info["flipped"] = bool(self.flipped)
            if info["landed"]:
                # +100 for a safe landing, up to +100 more for hitting the spot
                reward += 100 + 100 * max(0.0, 1 - info["miss"] / 6.0)
                if self.flip and self.flipped:
                    reward += self.FLIP_BONUS
            else:
                reward -= 100
                if self.flip:  # RL: a crash takes back the turning credit -- flips only pay if you land
                    reward -= self.FLIP_PROGRESS * min(abs(self.angle), 2 * np.pi)
        self.score += reward
        truncated = self.steps >= 1500
        if truncated and not terminated:      # RL: timed out -- report it like a failed attempt
            info.update(landed=False, miss=abs(self.x - self.TARGET_X), flipped=bool(self.flipped))
        return self._obs(), float(reward), terminated, truncated, info


gym.register(id="CarJump-v0", entry_point=CarJumpEnv, max_episode_steps=1500) \
    if "CarJump-v0" not in gym.registry else None


# ---------------- PILOT ----------------
def scripted_policy(obs):
    x, y, vy = obs[0] * 60, obs[1] * 6, obs[3] * 15
    if obs[7] == 1.0:                                   # on the road
        return np.array([1.0, 1.0 if x >= CarJumpEnv.LAUNCH_X else 0.0, 0.0], dtype=np.float32)
    a = np.arctan2(obs[4], obs[5])
    w = obs[6] * 5
    torque = float(np.clip(-(3.0 * a + 1.5 * w), -1, 1))          # keep level
    if x < CarJumpEnv.DESCEND_X:
        tv = float(np.clip((5.5 - y) * 1.2, -3.0, 4.5))           # climb, then cruise at 5.5 m
    else:
        tv = -float(np.clip(1.2 * y + 0.8, 0.8, 4.0))             # come down gently
    c = max(float(np.cos(a)), 0.5)
    thrust = float(np.clip((CarJumpEnv.G + 2.5 * (tv - vy)) / (CarJumpEnv.THRUST_ACC * c), 0, 1))
    return np.array([1.0, thrust, torque], dtype=np.float32)


def rollout(policy, seed=0, hold=70, target_x=None, flip=False):
    env = CarJumpEnv(target_x, flip)
    obs, _ = env.reset(seed=seed)
    fr, done, info = [], False, {}
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        fr.append(dict(x=env.x, y=env.y, a=env.angle, th=env.last_thrust, fuel=env.fuel,
                       s=float(np.hypot(env.vx, env.vy)), post=False, score=env.score))
        done = term or trunc
    landed = bool(info.get("landed", False))
    x, a, vx = env.x, wrap(env.angle), max(env.vx, 0.0)
    for _ in range(hold):
        if landed:
            vx *= 0.96
            x += vx * env.DT * 2
            a *= 0.85
            fr.append(dict(x=x, y=0.0, a=a, th=0.0, fuel=env.fuel, s=vx, post=True, score=env.score))
        else:
            fr.append(dict(x=env.x, y=max(env.y, 0.0), a=env.angle, th=0.0,
                           fuel=env.fuel, s=0.0, post=True, score=env.score))
    result = dict(landed=landed, score=env.score, miss=abs(env.x - env.TARGET_X), target_x=env.TARGET_X,
                  flipped=bool(env.flipped), turns=abs(env.angle) / (2 * np.pi))
    return fr, result


# ---------------- RENDERING ----------------
FIGW, FIGH, DPI = 8, 4.5, 80
W_M = 36.0
H_M = W_M * FIGH / FIGW
PX_PER_M = FIGW * DPI / W_M
Y_BOTTOM = -3.5
PIVOT = np.array([0.0, 0.6])
BODY = [(2.4, 0.45), (2.4, 0.95), (0.9, 1.05), (-0.4, 1.72), (-2.3, 1.38), (-2.3, 0.45)]
WINDOW = [(0.75, 1.12), (-0.35, 1.62), (-1.5, 1.44), (-1.5, 1.12)]
UNDER = [(-2.3, 0.45), (2.4, 0.45), (2.4, 0.64), (-2.3, 0.64)]
WHEELS = [(1.35, 0.45), (-1.35, 0.45)]
NOZZLES = [-1.7, -1.0, 1.0, 1.7]
LIGHTS = [(6, 12.5), (36, 13.5), (66, 12.5), (96, 13.0), (126, 12.5)]
SMOKE_RGB = (0.86, 0.88, 0.92)


def msize(d):
    return (np.asarray(d) * PX_PER_M * 72 / DPI) ** 2


def xf(P, a, x, y):
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return (np.asarray(P, float) - PIVOT) @ R.T + np.array([x, y + 0.6])


def glow(ax, x, y, r, rgb, a, z=7, n=7):
    for k in range(n):
        ax.add_patch(Circle((x, y), r * (1 - k / n), fc=(*rgb, a / n * 1.6), ec="none", zorder=z))


class Smoke:
    def __init__(self):
        self.p = []

    def add(self, x, y, vx, vy, life, size0, grow, rgb, a0, buoy=0.4, drag=1.0):
        self.p.append([x, y, vx, vy, 0.0, life, size0, grow, rgb, a0, buoy, drag])

    def step(self, dt):
        keep = []
        for q in self.p:
            q[4] += dt
            if q[4] >= q[5]:
                continue
            k = np.exp(-q[11] * dt)
            q[2] *= k
            q[3] = q[3] * k + q[10] * dt
            q[0] += q[2] * dt
            q[1] += q[3] * dt
            if q[1] < 0.05 and q[3] < 0:
                q[1], q[3] = 0.05, q[3] * -0.2
            keep.append(q)
        self.p = keep[-800:]

    def draw(self, ax):
        if not self.p:
            return
        P = self.p
        x = np.array([q[0] for q in P]); y = np.array([q[1] for q in P])
        age = np.array([q[4] for q in P]); life = np.array([q[5] for q in P])
        size = np.array([q[6] for q in P]) + np.array([q[7] for q in P]) * age
        a0 = np.array([q[9] for q in P])
        rgb = np.array([q[8] for q in P])
        a = a0 * (1 - age / life) ** 1.5 * np.minimum(1, age / 0.08)
        for k, w in ((1.0, 0.55), (1.8, 0.30), (2.8, 0.18)):
            col = np.c_[rgb, np.clip(a * w, 0, 1)]
            ax.scatter(x, y, s=msize(size * k), c=col, linewidths=0, zorder=4)


def emit(sm, f, rng):
    x, y, a, th = f["x"], f["y"], f["a"], f["th"]
    up = np.array([-np.sin(a), np.cos(a)])
    if th > 0.05:
        for i, lx in enumerate(NOZZLES):
            n = xf([[lx, 0.3]], a, x, y)[0]
            for _ in range(int(1 + 3 * th)):
                v = -up * (5 + 5 * th) + rng.normal(0, 0.8, 2)
                sm.add(n[0], n[1], v[0], v[1], rng.uniform(1.0, 1.8), 0.3, 2.0, SMOKE_RGB, 0.25)
            if i % 2 == 0 and n[1] < 9 and up[1] > 0.3:          # blast hitting the road
                gx = n[0] - up[0] * (n[1] / up[1])
                for _ in range(3):
                    sg = rng.choice([-1, 1])
                    sm.add(gx + rng.normal(0, 0.5), 0.2, sg * rng.uniform(3, 9),
                           rng.uniform(0.3, 1.8), rng.uniform(1.5, 2.5), 0.7, 2.2,
                           SMOKE_RGB, 0.30, buoy=0.25, drag=0.9)
    elif f["s"] > 2 and y < 0.2:
        n = xf([[-1.35, 0.0]], a, x, y)[0]
        sm.add(n[0], n[1], -f["s"] * 0.15 + rng.normal(0, .5), rng.uniform(0.3, 1.5),
               0.9, 0.3, 1.6, (0.6, 0.6, 0.62), 0.15)


def crash_burst(sm, f, rng):
    for i in range(80):
        ang = rng.uniform(0, np.pi)
        v = rng.uniform(2, 9) * np.array([np.cos(ang), np.sin(ang)])
        if i < 25:
            sm.add(f["x"], f["y"] + 0.8, v[0], v[1], 0.5, 0.4, 2.5, (1.0, 0.55, 0.15), 0.5, 0.2, 1.5)
        else:
            sm.add(f["x"], f["y"] + 0.8, v[0] * .6, v[1] * .6, rng.uniform(1.2, 2.2),
                   0.6, 3.0, (0.5, 0.5, 0.52), 0.35, 0.5, 1.0)


def draw_wheel(ax, cx, cy, r, rot, z):
    ax.add_patch(Circle((cx, cy), r, fc="#0b0c0e", ec="#2b2d33", lw=0.8, zorder=z))
    ax.add_patch(Circle((cx, cy), r * 0.5, fc="#6c7078", ec="none", zorder=z + 0.05))
    ax.plot([cx, cx + r * 0.5 * np.cos(rot)], [cy, cy + r * 0.5 * np.sin(rot)],
            color="#1b1c20", lw=1, zorder=z + 0.1)


def draw_vehicle(ax, x0, x1, h, kind, color):
    if kind == "truck":
        ax.add_patch(Rectangle((x0, 0.55), x1 - x0 - 1.7, h - 0.55, fc=color, ec="#c9ccd2", lw=.8, zorder=3))
        ax.add_patch(Polygon([(x1 - 1.7, .55), (x1, .55), (x1, 2.1), (x1 - .8, 2.9), (x1 - 1.7, 2.9)],
                             fc=color, ec="#c9ccd2", lw=.8, zorder=3))
        ax.add_patch(Polygon([(x1 - 1.45, 1.5), (x1 - .2, 1.5), (x1 - .2, 2.05), (x1 - .85, 2.7), (x1 - 1.45, 2.7)],
                             fc="#15171c", zorder=3.1))
        for wx in (x0 + .9, x0 + 1.9, x1 - .9):
            draw_wheel(ax, wx, .5, .5, 0.3, 3.2)
        glow(ax, x1, 1.0, .5, (1, .95, .7), .5, z=3.4)
    elif kind == "cyber":
        ax.add_patch(Polygon([(x0, .5), (x0, 1.5), (x0 + 1.7, 1.55), (x0 + 2.4, 1.95),
                              (x1 - .1, 1.2), (x1, 1.0), (x1, .5)],
                             fc=color, ec="#eef0f4", lw=.9, zorder=3))
        ax.add_patch(Polygon([(x0 + 2.5, 1.85), (x1 - .6, 1.22), (x1 - 2.0, 1.22), (x0 + 2.3, 1.58)],
                             fc="#12141a", zorder=3.1))
        ax.plot([x0 + 1.7, x1 - .1], [1.55, 1.2], color="w", alpha=.35, lw=1, zorder=3.2)
        for wx in (x0 + 1.2, x1 - 1.2):
            draw_wheel(ax, wx, .5, .5, 0.4, 3.2)
        ax.plot([x1 - .45, x1], [1.03, 1.03], color="white", lw=2.5, zorder=3.3)   # light bar
        glow(ax, x1, 1.03, .6, (1, 1, 1), .6, z=3.4)
        ax.plot([x0, x0], [1.3, 1.42], color="#ff2a2a", lw=2.5, zorder=3.3)
    else:
        ax.add_patch(Polygon([(x0, .4), (x0, .75), (x0 + .8, .85), (x0 + 1.3, h), (x1 - 1.2, h),
                              (x1 - .5, .85), (x1, .75), (x1, .4)], fc=color, ec="#d0d3d8", lw=.7, zorder=3))
        ax.add_patch(Polygon([(x0 + 1.4, .95), (x0 + 1.5, h - .1), (x1 - 1.3, h - .1), (x1 - .8, .95)],
                             fc="#12141a", zorder=3.1))
        for wx in (x0 + .9, x1 - .9):
            draw_wheel(ax, wx, .4, .4, 0.6, 3.2)


def draw_car(ax, f, wr, rng):
    x, y, a, th = f["x"], f["y"], f["a"], f["th"]
    nose = xf([[2.4, 0.8]], a, x, y)[0]
    bd = np.array([np.cos(a - 0.12), np.sin(a - 0.12)])
    bp = np.array([-bd[1], bd[0]])
    for L, wd, al in ((9, 2.6, .05), (6, 1.6, .06), (3.5, .8, .07)):
        ax.add_patch(Polygon([nose, nose + bd * L + bp * wd, nose + bd * L - bp * wd],
                             fc=(1, 1, .92, al), ec="none", zorder=5))
    d = np.array([np.sin(a), -np.cos(a)])
    p = np.array([np.cos(a), np.sin(a)])
    if th > 0.05:                                    # vapor jets: pale streaks, no flame
        for lx in NOZZLES:
            n = xf([[lx, 0.3]], a, x, y)[0]
            L = (1.2 + 3.8 * th) * rng.uniform(0.9, 1.1)
            for w, ln, col in ((.16, 1.0, (.80, .88, 1, .22)), (.09, .8, (.92, .96, 1, .35)),
                               (.04, .55, (1, 1, 1, .5))):
                ax.add_patch(Polygon([n + p * w, n + d * L * ln, n - p * w],
                                     fc=col, ec="none", zorder=5.5))
    ax.add_patch(Polygon(xf(BODY, a, x, y), fc="#ececef", ec="#ffffff", lw=.8, zorder=6))
    ax.add_patch(Polygon(xf(UNDER, a, x, y), fc="#2a2c31", ec="none", zorder=6.1))
    ax.add_patch(Polygon(xf(WINDOW, a, x, y), fc="#0c0e13", ec="#3a3d45", lw=.6, zorder=6.2))
    for lx in NOZZLES:                               # nozzle housings under the car
        ax.add_patch(Polygon(xf([[lx - .15, .45], [lx + .15, .45], [lx + .12, .28], [lx - .12, .28]], a, x, y),
                             fc="#1a1b1f", ec="#3a3d45", lw=.5, zorder=6.15))
    for k, (wx, wy) in enumerate(WHEELS):
        c = xf([[wx, wy]], a, x, y)[0]
        draw_wheel(ax, c[0], c[1], .45, wr + k, 6.3)
    glow(ax, nose[0], nose[1], .55, (1, 1, .92), .9)
    tail = xf([[-2.3, 0.95]], a, x, y)[0]
    glow(ax, tail[0], tail[1], .35, (1, .1, .1), .8)


def draw_scene(ax, f, cam, wr, sm, rng, landed, last, target_x=CarJumpEnv.TARGET_X, label=None, miss=None,
               extras=False, reason=None):
    E = CarJumpEnv
    ax.cla()
    ax.set_xlim(cam - 0.4 * W_M, cam + 0.6 * W_M)
    ax.set_ylim(Y_BOTTOM, Y_BOTTOM + H_M)
    ax.axis("off")
    for j in range(20):
        t = j / 19
        c = np.array([.045, .055, .09]) * (1 - t) + np.array([.01, .012, .025]) * t
        ax.add_patch(Rectangle((0, 0.2 + 0.8 * j / 20), 1, 0.8 / 20 + 0.003, transform=ax.transAxes,
                               fc=c, ec="none", zorder=0))
    for lx, ly in LIGHTS:
        ax.plot([lx, lx], [0, ly], color="#1a1c22", lw=2, zorder=1)
        ax.add_patch(Polygon([(lx, ly), (lx - 7, 0), (lx + 7, 0)], fc=(1, 1, .95, .03), ec="none", zorder=1))
        glow(ax, lx, ly, 1.6, (1, 1, .95), .55, z=2)
    ax.add_patch(Rectangle((cam - 25, Y_BOTTOM), 70, -Y_BOTTOM, fc="#0e0f13", ec="none", zorder=1))
    ax.plot([cam - 25, cam + 45], [0, 0], color="#2a2d35", lw=1, zorder=1.5)
    for xx in range(int(cam) - 25, int(cam) + 30):
        if xx % 6 == 0:
            ax.plot([xx, xx + 3], [-1.8, -1.8], color="w", alpha=.4, lw=1.3, zorder=1.5)
    ax.plot([E.LAUNCH_X, E.LAUNCH_X], [0.05, -0.9], color=(1, .75, .24, .8), lw=2, zorder=3.5)  # jets-on line
    for lw, al in ((14, .07), (8, .15), (2.5, 1.0)):
        ax.plot([27, 60], [0.1, 0.1], color=(1, .1, .1, al), lw=lw, zorder=3.5)
    for lw, al in ((10, .1), (2.5, .9)):
        ax.plot([E.LAND_X, E.LAND_X + 45], [0.08, 0.08], color=(.24, .86, .52, al), lw=lw, zorder=3.5)
    # RL: the landing spot -- a bracket on the green strip (extras only; off = the PDF's look)
    for dx in ((-2.6, 2.6) if extras else ()):
        ax.plot([target_x + dx, target_x + dx], [0.08, -0.7], color=(.24, .86, .52, .9), lw=2, zorder=3.6)
    if extras:
        ax.plot([target_x - 2.6, target_x + 2.6], [-0.7, -0.7], color=(.24, .86, .52, .9), lw=2, zorder=3.6)
        ax.text(target_x, -1.25, "LAND HERE", ha="center", va="top", color=(.24, .86, .52),
                fontsize=7, family="monospace", zorder=3.6)
    for x0, x1, h, kind, col in E.OBSTACLES:
        draw_vehicle(ax, x0, x1, h, kind, col)
    sm.draw(ax)
    draw_car(ax, f, wr, rng)
    ax.text(0.97, 0.05, f"ALTITUDE\n{max(f['y'], 0) * 3.281:.1f} FT", transform=ax.transAxes,
            ha="right", va="bottom", color="white", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=(0, 0, 0, .45), ec="white", lw=1), zorder=20)
    ax.add_patch(Rectangle((0.03, 0.92), 0.18, 0.03, transform=ax.transAxes, fc=(0, 0, 0, .5),
                           ec="white", lw=1, zorder=20))
    fu = f["fuel"]
    ax.add_patch(Rectangle((0.03, 0.92), 0.18 * fu, 0.03, transform=ax.transAxes,
                           fc=(1 - fu, 0.8 * fu + .1, .2), ec="none", zorder=21))
    ax.text(0.03, 0.96, "JET FUEL", transform=ax.transAxes, color="white", fontsize=8,
            family="monospace", zorder=22)
    # RL: live score, and which attempt / model this is
    if extras:
        ax.text(0.03, 0.86, f"SCORE {f.get('score', 0.0):7.1f}", transform=ax.transAxes, color="white",
                fontsize=9, family="monospace", va="top", zorder=22)
    if extras and f.get("phys"):   # per-frame physics readout (replays)
        ax.text(0.03, 0.80, f["phys"], transform=ax.transAxes, color=(.75, .8, .9), fontsize=7,
                family="monospace", va="top", linespacing=1.3, zorder=22)
    if extras and label:
        ax.text(0.97, 0.96, label, transform=ax.transAxes, ha="right", va="top", color="white",
                fontsize=8, family="monospace", zorder=22)
    if last:
        ax.text(0.5, 0.88, "LANDED" if landed else "CRASHED", transform=ax.transAxes, ha="center",
                color=(.24, .86, .52) if landed else (1, .3, .25), fontsize=18,
                family="monospace", weight="bold", zorder=22)
        if extras and landed and miss is not None:
            ax.text(0.5, 0.80, f"{miss:.1f} m from the spot" + ("  - ON THE SPOT" if miss < 1 else ""),
                    transform=ax.transAxes, ha="center", color="white", fontsize=10, family="monospace", zorder=22)
        if extras and not landed and reason:
            ax.text(0.5, 0.80, reason, transform=ax.transAxes, ha="center",
                    color="white", fontsize=10, family="monospace", zorder=22)


H_PX, W_PX = int(FIGH * DPI), int(FIGW * DPI)
_yy, _xx = np.mgrid[0:H_PX, 0:W_PX]
_r2 = ((_xx - W_PX / 2) / (W_PX / 2)) ** 2 + ((_yy - H_PX / 2) / (H_PX / 2)) ** 2
VIGNETTE = (1 - 0.35 * np.clip(_r2, 0, 1.4))[..., None].astype(np.float32)


def post(img, rng):
    arr = img.astype(np.float32)
    bright = np.where(arr.mean(2, keepdims=True) > 170, arr, 0).astype(np.uint8)
    bloom = np.asarray(PILImage.fromarray(bright).filter(ImageFilter.GaussianBlur(7)), dtype=np.float32)
    out = (arr + 0.8 * bloom) * VIGNETTE
    out += rng.normal(0, 3.5, (H_PX, W_PX, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def render_frames(frames, result, stride=2, label=None, extras=False):
    """The PDF's make_gif loop, returning the images so several attempts can share one GIF."""
    landed = result["landed"]
    rng = np.random.default_rng(3)
    dt = CarJumpEnv.DT * stride
    fr = frames[::stride]
    burst_i = next((i for i, f in enumerate(fr) if f["post"]), None)
    fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI)
    fig.patch.set_facecolor("#03040a")
    ax = fig.add_axes([0, 0, 1, 1])
    sm, cam, wr, imgs = Smoke(), None, 0.0, []
    for i, f in enumerate(fr):
        emit(sm, f, rng)
        if (not landed) and i == burst_i:
            crash_burst(sm, f, rng)
        sm.step(dt)
        wr += f["s"] * dt / 0.45
        tgt = f["x"] + 6
        cam = tgt if cam is None else cam + (tgt - cam) * 0.18
        last = (f["post"] and i >= burst_i) if extras else i == len(fr) - 1   # PDF: result on the last frame
        draw_scene(ax, f, cam, wr, sm, rng, landed, last,
                   target_x=result["target_x"], label=label, miss=result["miss"], extras=extras,
                   reason=result.get("reason"))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        imgs.append(PILImage.fromarray(post(rgb, rng)))
    plt.close(fig)
    return imgs, int(dt * 1000)


def save_gif(imgs, ms, path, width=480, colors=64):
    """Writes path (GIF) and the same animation as .mp4 at full quality.

    The film grain makes every frame unique, so GIFs stay big; the MP4 is the small one to share.
    """
    import imageio.v2 as imageio
    imageio.mimwrite(path.rsplit(".", 1)[0] + ".mp4", [np.asarray(im) for im in imgs],
                     fps=1000 / ms, codec="libx264", quality=8, macro_block_size=8)
    imgs = [im.resize((width, round(im.height * width / im.width)), PILImage.LANCZOS) for im in imgs]
    palette = imgs[len(imgs) // 2].quantize(colors=colors, method=PILImage.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=palette, dither=PILImage.Dither.NONE) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)


def make_gif(frames, result, path="car_jump.gif", stride=2, label=None):
    imgs, ms = render_frames(frames, result, stride, label)
    save_gif(imgs, ms, path)
    print(f"Saved {path} ({len(imgs)} frames) - {'LANDED' if result['landed'] else 'CRASHED'}")


if __name__ == "__main__":
    frames, result = rollout(scripted_policy, seed=1)
    make_gif(frames, result, "car_jump.gif", label="SCRIPTED PILOT")
    print(result)
