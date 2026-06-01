#!/usr/bin/env python
"""One-click driver for the RoboTwin notebook.

RoboTwin ships no pretrained policies — but every task has a scripted *expert planner*
(curobo + mplib) that solves it. That planner is what generates the 100k-trajectory
dataset, and it's the most compelling thing to watch: a dual-arm robot actually doing
the task, no training required.

`preview` reuses the upstream `script/collect_data.py` seed-search phase, which runs the
expert `play_once()` per episode. We just flip `render_freq>0` (pops the SAPIEN viewer)
and `collect_data:false` (skip the heavy HDF5 write) via a generated throwaway config.

Usage:
    python scripts/robotwin_demo.py list
    python scripts/robotwin_demo.py preview handover_block --gpu 0
    python scripts/robotwin_demo.py preview beat_block_hammer --config demo_randomized --episodes 2 --gpu 0
    python scripts/robotwin_demo.py collect stack_blocks_two --config demo_clean --episodes 50 --gpu 0
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(REPO_ROOT, "dependencies", "RoboTwin")

# envs/*.py that are not tasks
_NON_TASKS = {"_base_task", "_GLOBAL_CONFIGS", "__init__"}


def list_tasks():
    envs_dir = os.path.join(RT, "envs")
    tasks = sorted(
        f[:-3]
        for f in os.listdir(envs_dir)
        if f.endswith(".py") and f[:-3] not in _NON_TASKS and not f.startswith("__")
    )
    print(f"# {len(tasks)} RoboTwin tasks\n")
    for i, t in enumerate(tasks):
        print(f"  {t:<28}", end="" if i % 2 == 0 else "\n")
    if len(tasks) % 2:
        print()
    return tasks


def _write_preview_config(base_config, *, render_freq, collect_data, episodes):
    """Derive a throwaway config from a base one, with preview overrides."""
    base_path = os.path.join(RT, "task_config", f"{base_config}.yml")
    if not os.path.isfile(base_path):
        sys.exit(f"❌ base config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["render_freq"] = render_freq          # >0 → SAPIEN viewer pops up
    cfg["collect_data"] = collect_data        # False → skip heavy HDF5 write for previews
    cfg["use_seed"] = False                   # always re-search → re-plays the expert live
    cfg["episode_num"] = episodes

    out_name = "_preview" if not collect_data else f"_run_{base_config}"
    out_path = os.path.join(RT, "task_config", f"{out_name}.yml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_name


def _run_collect(task, config_name, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS="ignore::UserWarning")
    cmd = [sys.executable, "script/collect_data.py", task, config_name]
    print(f"▶ (cwd={RT})  CUDA_VISIBLE_DEVICES={gpu} python {' '.join(cmd[1:])}\n")
    return subprocess.call(cmd, cwd=RT, env=env)


def cmd_preview(args):
    cfg_name = _write_preview_config(
        args.config, render_freq=args.render_freq, collect_data=False, episodes=args.episodes
    )
    # wipe any prior preview output so the expert re-plays live every time
    shutil.rmtree(os.path.join(RT, "data", args.task, cfg_name), ignore_errors=True)
    rc = _run_collect(args.task, cfg_name, args.gpu)
    shutil.rmtree(os.path.join(RT, "data", args.task, cfg_name), ignore_errors=True)
    sys.exit(rc)


def cmd_collect(args):
    cfg_name = _write_preview_config(
        args.config, render_freq=0, collect_data=True, episodes=args.episodes
    )
    sys.exit(_run_collect(args.task, cfg_name, args.gpu))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all RoboTwin tasks")

    pp = sub.add_parser("preview", help="watch the expert planner solve a task (SAPIEN viewer)")
    pp.add_argument("task")
    pp.add_argument("--config", default="demo_clean", help="base config (demo_clean / demo_randomized)")
    pp.add_argument("--episodes", type=int, default=3, help="how many successful episodes to play")
    pp.add_argument("--render-freq", type=int, default=8, help="viewer render every N control steps (>0)")
    pp.add_argument("--gpu", type=int, default=0)

    pc = sub.add_parser("collect", help="generate demonstration data (headless, writes HDF5)")
    pc.add_argument("task")
    pc.add_argument("--config", default="demo_clean")
    pc.add_argument("--episodes", type=int, default=50)
    pc.add_argument("--gpu", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "list":
        list_tasks()
    elif args.cmd == "preview":
        cmd_preview(args)
    elif args.cmd == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
