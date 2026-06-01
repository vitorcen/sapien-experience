# G1 PlaceAppleInBowl — PPO checkpoint

强化学习(PPO)训练的 Unitree G1 双臂"把苹果放进碗"策略。**纯奖励驱动,无示范数据集、无脚本规则。**

_Reinforcement-learning (PPO) policy for Unitree G1 placing an apple into a bowl. Reward-driven only — no demonstration dataset, no scripted rules._

| 项 / Item | 值 / Value |
|---|---|
| 任务 / Task | `UnitreeG1PlaceAppleInBowl-v1` (ManiSkill 3) |
| 算法 / Algo | PPO (`examples/baselines/ppo/ppo.py`), state-based |
| 网络 / Net | MLP actor+critic, obs 74 → act 25, ~0.3M params |
| 训练 / Training | 50M steps, 1024 envs, ~80 min on RTX 4090 |
| 结果 / Result | eval return 3 → **50**(学会双臂伸手抓取),但确定性 eval **0% 干净放置**——卡在 grasp-release 局部最优 |

文件 / File: `final_ckpt.pt` (1.2 MB) — `agent.state_dict()` from ppo.py.

## 复现 GUI eval / Reproduce live GUI eval

```bash
DISPLAY=:0 conda run -n maniskill python scripts/g1_eval_gui.py \
    --ckpt checkpoints/g1_placeapple/final_ckpt.pt --episodes 5
```

详见 [`../../ManiSkill.ipynb`](../../ManiSkill.ipynb) §7。31 个中间 checkpoint(供 sweep)未入库,在 `dependencies/ManiSkill/examples/baselines/ppo/runs/g1_apple/`(被 .gitignore 排除)。
