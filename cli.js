#!/usr/bin/env node
// car-jump: npm front end for the Python project.
// It keeps a private Python environment in .venv/ and runs the scripts there.
//
//   npx car-jump setup         one-time install (Python 3.10+ needed)
//   npx car-jump demo          trained AI: 20-attempt test + car_jump_ai.gif
//   npx car-jump watch         train with the live animation of every attempt
//   npx car-jump help          everything else

"use strict";
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const VENV = path.join(ROOT, ".venv");
const IS_WIN = process.platform === "win32";
const VENV_PY = path.join(VENV, IS_WIN ? "Scripts/python.exe" : "bin/python");

const COMMANDS = {
  setup: { help: "create .venv and install the Python packages (once)" },
  demo: { script: "make_jump_gifs.py", help: "test the included pretrained AI over 20 attempts and render car_jump_ai.gif" },
  test: { script: "make_jump_gifs.py", args: ["--test-only"], help: "only the 20-attempt test of the trained AI" },
  train: { script: "train_jump.py", help: "train from scratch, printing every attempt (obs, actions, reward)" },
  watch: { script: "train_jump.py", args: ["--watch"], help: "train + live animation of every attempt" },
  flip: { script: "train_jump.py", args: ["--flip", "--steps", "4000000", "--watch"], help: "teach a mid-air flip, with live animation" },
  view: { script: "live_view.py", help: "replay a run's attempts as a live animation" },
  replay: { script: "replay_attempts.py", help: "render learning_replay.mp4 + logs/learning_curve.png from the logs" },
  scripted: { script: "car_jump.py", help: "the PDF's original scripted pilot -> car_jump.gif" },
};

function run(cmd, args) {
  const r = spawnSync(cmd, args, { cwd: ROOT, stdio: "inherit" });
  if (r.error) return { ok: false, error: r.error };
  return { ok: r.status === 0, status: r.status };
}

function findPython() {
  const candidates = IS_WIN ? ["py", "python", "python3"] : ["python3", "python"];
  for (const c of candidates) {
    const r = spawnSync(c, ["-c", "import sys; print(sys.version_info >= (3, 10))"], { encoding: "utf8" });
    if (r.status === 0 && r.stdout.trim() === "True") return c;
  }
  return null;
}

function setup() {
  if (!fs.existsSync(VENV_PY)) {
    const py = findPython();
    if (!py) {
      console.error("Python 3.10 or newer is needed: https://www.python.org/downloads/");
      process.exit(1);
    }
    console.log(`Creating .venv with ${py} ...`);
    if (!run(py, ["-m", "venv", VENV]).ok) process.exit(1);
  }
  console.log("Installing Python packages (first time takes a few minutes: PyTorch is large) ...");
  const steps = [
    ["-m", "pip", "install", "--upgrade", "pip"],
    ["-m", "pip", "install", "-r", "requirements.txt"],
  ];
  for (const s of steps) if (!run(VENV_PY, s).ok) process.exit(1);
  console.log("\nReady. Try:  npm run demo   or   npm run watch");
}

function help() {
  console.log("car-jump: teach a jet-pack car to jump parked cars and land on a spot (Gymnasium + PPO)\n");
  console.log("usage: npm run <command> [-- extra args]   or   npx car-jump <command> [extra args]\n");
  for (const [name, c] of Object.entries(COMMANDS)) console.log(`  ${name.padEnd(9)} ${c.help}`);
  console.log("\nexamples:");
  console.log("  npm run train -- --target 80          land on a different spot");
  console.log("  npm run view -- --from-attempt 300    re-watch from attempt 300");
  console.log("  npm run demo -- --name jump_t67_s0      test + GIFs of your own training run");
}

const [name = "help", ...extra] = process.argv.slice(2);
if (name === "setup") setup();
else if (!COMMANDS[name]) help();
else {
  if (!fs.existsSync(VENV_PY)) {
    console.log("First run: setting up Python ...");
    setup();
  }
  const c = COMMANDS[name];
  const r = run(VENV_PY, [c.script, ...(c.args || []), ...extra]);
  process.exit(r.ok ? 0 : r.status || 1);
}
