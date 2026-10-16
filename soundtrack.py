"""An original soundtrack for make_video.py, synthesised from scratch (no samples, no
copyright): a beat with chords that go from tense (crashes) to uplifting (landings), plus
sound effects timed to what happens on screen -- jet hiss, crash booms, touchdown thuds,
and a chime on the perfect landing.
"""
import wave

import numpy as np

SR = 44100
BPM = 112
BEAT = 60 / BPM

# chords as MIDI notes; tense minor while it crashes, brighter once it starts landing
TENSE = [[57, 60, 64], [53, 57, 60], [57, 60, 64], [52, 56, 59]]      # Am  F  Am  E
BRIGHT = [[48, 52, 55], [55, 59, 62], [57, 60, 64], [53, 57, 60]]     # C   G  Am  F
RESOLVE = [[48, 52, 55, 60]]                                           # C (held)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def smooth(x, n):
    """Crude low-pass: moving average over n samples."""
    return np.convolve(x, np.ones(n) / n, mode="same")


def tone(freq, dur, harmonics=6, detune=0.0):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        out += np.sin(2 * np.pi * freq * k * (1 + detune) * t) / k ** 1.4
    return out


def envelope(n, attack, release):
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    if a:
        env[:a] = np.linspace(0, 1, min(a, n))[: len(env[:a])]
    if r:
        env[-r:] *= np.linspace(1, 0, min(r, n))
    return env


def add(buf, sig, at):
    i = int(at * SR)
    if i < 0:  # starts before the video: keep only the part after 0 s
        sig, i = sig[-i:], 0
    if i >= len(buf) or not len(sig):
        return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += sig


def pad(buf, start, end, progression, level):
    bar = 4 * BEAT
    t, k = start, 0
    while t < end:
        dur = min(bar, end - t)
        chord = progression[k % len(progression)]
        sig = sum(tone(hz(m), dur, detune=d) for m in chord for d in (0.0, 0.003))
        sig *= envelope(len(sig), 0.35, 0.5) * level / (2 * len(chord))
        add(buf, sig, t)
        add(buf, 0.5 * level * np.sin(2 * np.pi * hz(chord[0] - 24) * np.arange(len(sig)) / SR)
            * envelope(len(sig), 0.05, 0.3), t)  # sub bass
        t += bar
        k += 1


def drums(buf, start, end, level, rng, hats=True):
    t = start
    beat_i = 0
    while t < end:
        n = int(0.35 * SR)
        tt = np.arange(n) / SR
        kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
        add(buf, kick * level, t)
        if hats:
            m = int(0.05 * SR)
            hat = rng.normal(0, 1, m)
            hat = (hat - smooth(hat, 8)) * np.exp(-np.arange(m) / SR * 80)
            add(buf, hat * level * 0.25, t + BEAT / 2)
        if beat_i % 2 == 1:  # clap on 2 and 4
            m = int(0.12 * SR)
            clap = smooth(rng.normal(0, 1, m), 3) * np.exp(-np.arange(m) / SR * 30)
            add(buf, clap * level * 0.35, t)
        t += BEAT
        beat_i += 1


def boom(buf, at, rng, level=1.0):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    low = np.sin(2 * np.pi * (35 + 70 * np.exp(-t * 6)) * t) * np.exp(-t * 3)
    crack = smooth(rng.normal(0, 1, n), 6) * np.exp(-t * 9)
    rumble = smooth(rng.normal(0, 1, n), 60) * np.exp(-t * 2.5) * 4
    add(buf, level * (0.9 * low + 0.5 * crack + 0.5 * rumble), at)


def thud(buf, at, level=0.6):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    add(buf, level * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 18), at)


def chime(buf, at, level=0.35):
    for i, m in enumerate((72, 76, 79, 84)):   # C E G C, rising
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        sig = (np.sin(2 * np.pi * hz(m) * t) + 0.3 * np.sin(2 * np.pi * hz(m) * 2 * t)) * np.exp(-t * 2.2)
        add(buf, level * sig, at + i * 0.09)


def whoosh(buf, at, rng, level=0.25):
    n = int(0.45 * SR)
    noise = rng.normal(0, 1, n)
    sig = (smooth(noise, 4) - smooth(noise, 40)) * np.sin(np.linspace(0, np.pi, n)) ** 2
    add(buf, level * sig, at - 0.2)


def jets(buf, thrust, fps, rng, level=0.22):
    """Hiss that follows the jets' thrust, frame by frame."""
    per = SR / fps
    env = np.repeat(np.asarray(thrust, float), int(np.ceil(per)))[: len(buf)]
    env = smooth(env, int(0.03 * SR))
    noise = rng.normal(0, 1, len(env))
    hiss = smooth(noise, 3) - smooth(noise, 25) + 0.4 * smooth(noise, 120)
    buf[: len(env)] += level * env * hiss


def make_soundtrack(path, duration, fps, scenes, end_card_start):
    """scenes: list of dicts with start (s), thrust (per frame), impact (s or None),
    touchdown (s or None), perfect (bool). end_card_start in seconds."""
    rng = np.random.default_rng(7)
    music = np.zeros(int(duration * SR) + SR)
    sfx = np.zeros_like(music)

    first_landing = next((s["start"] for s in scenes if s["touchdown"] is not None), end_card_start)
    pad(music, 0.0, first_landing, TENSE, 0.55)
    pad(music, first_landing, end_card_start, BRIGHT, 0.6)
    pad(music, end_card_start, duration + 1, RESOLVE, 0.7)
    drums(music, 1.0, first_landing, 0.55, rng, hats=False)        # sparse while it crashes
    drums(music, first_landing, end_card_start, 0.65, rng)          # full beat once it lands
    drums(music, end_card_start, end_card_start + 4 * BEAT, 0.5, rng)

    thrust = []
    for s in scenes:
        whoosh(sfx, s["start"], rng)
        thrust += list(s["thrust"])
        if s["impact"] is not None:
            boom(sfx, s["impact"], rng)
        if s["touchdown"] is not None:
            thud(sfx, s["touchdown"])
            if s["perfect"]:
                chime(sfx, s["touchdown"] + 0.15)
    jets(sfx, thrust, fps, rng)
    chime(sfx, end_card_start + 0.3, level=0.25)

    # duck the music under each boom so it hits
    duck = np.ones_like(music)
    for s in scenes:
        if s["impact"] is not None:
            i = int(s["impact"] * SR)
            n = int(0.8 * SR)
            duck[i:i + n] *= 1 - 0.6 * np.exp(-np.arange(min(n, len(duck) - i)) / SR * 4)
    mix = 0.55 * music * duck + sfx
    mix = np.tanh(2.0 * mix / (np.max(np.abs(mix)) + 1e-9)) * 0.89
    mix = mix[: int(duration * SR)]
    fade = int(0.8 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    stereo = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())
