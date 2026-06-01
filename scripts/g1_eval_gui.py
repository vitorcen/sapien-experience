#!/usr/bin/env python
"""Live GUI eval of the trained UnitreeG1PlaceAppleInBowl PPO policy.

Loads a ppo.py checkpoint, runs the deterministic actor in render_mode="human"
so the SAPIEN viewer shows G1 reaching/grasping/placing the apple live.

Usage (needs DISPLAY):
    DISPLAY=:0 conda run -n maniskill python scripts/g1_eval_gui.py \
        --ckpt dependencies/ManiSkill/examples/baselines/ppo/runs/g1_apple/final_ckpt.pt \
        --episodes 5
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Same architecture as examples/baselines/ppo/ppo.py."""

    def __init__(self, obs_n, act_n):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_n, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_n, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_n), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_n) * -0.5)

    @torch.no_grad()
    def act(self, x):  # deterministic
        return self.actor_mean(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-id", default="UnitreeG1PlaceAppleInBowl-v1")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_kwargs = dict(obs_mode="state", render_mode="human",
                      sim_backend="physx_cuda", control_mode="pd_joint_delta_pos")
    env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    obs_n = int(np.array(env.single_observation_space.shape).prod())
    act_n = int(np.prod(env.single_action_space.shape))
    agent = Agent(obs_n, act_n).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()
    print(f"loaded {args.ckpt}  (obs={obs_n}, act={act_n})", flush=True)

    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        env.render()
        ep_success = False
        for _ in range(args.max_steps):
            obs, rew, term, trunc, info = env.step(agent.act(obs))
            env.render()
            if "success" in info and bool(info["success"].any()):
                ep_success = True
        successes += int(ep_success)
        print(f"episode {ep}: {'✅ success' if ep_success else '— reached/grasped, no clean place'}", flush=True)
    print(f"\nGUI eval: {successes}/{args.episodes} success", flush=True)
    env.close()


if __name__ == "__main__":
    main()
