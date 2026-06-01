#!/usr/bin/env python
"""Tiny introspection helper for the ManiSkill notebook.

The actual demos are first-party ManiSkill entry points called directly from the
notebook (`python -m mani_skill.examples.demo_random_action ...`). This script only
covers the two things that are annoying to inline: listing registered envs/robots,
and a one-glance status print.

Usage:
    python scripts/maniskill_demo.py status
    python scripts/maniskill_demo.py list            # registered task env ids
    python scripts/maniskill_demo.py list --robots   # registered robot uids
"""
import argparse
import sys


def cmd_status(_args):
    import mani_skill
    import sapien
    import torch

    print(f"mani_skill : {mani_skill.__version__}")
    print(f"sapien     : {sapien.__version__}")
    print(f"torch      : {torch.__version__}  (cuda={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"gpu        : {torch.cuda.get_device_name(0)}")


def cmd_list(args):
    import mani_skill.envs  # noqa: F401  registers all envs
    from mani_skill.utils.registration import REGISTERED_ENVS

    if args.robots:
        import mani_skill.agents.robots  # noqa: F401  registers all agents
        from mani_skill.agents.registration import REGISTERED_AGENTS

        names = sorted(REGISTERED_AGENTS.keys())
        print(f"# {len(names)} registered robot uids\n")
        for n in names:
            print(f"  {n}")
        return

    ids = sorted(REGISTERED_ENVS.keys())
    print(f"# {len(ids)} registered ManiSkill task env ids\n")
    for env_id in ids:
        spec = REGISTERED_ENVS[env_id]
        steps = getattr(spec, "max_episode_steps", None)
        print(f"  {env_id:<34} max_steps={steps}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print versions / GPU")

    pl = sub.add_parser("list", help="list registered envs (or --robots)")
    pl.add_argument("--robots", action="store_true", help="list robot uids instead of tasks")

    args = p.parse_args()
    {"status": cmd_status, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
