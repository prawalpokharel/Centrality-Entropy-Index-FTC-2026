"""
PPO baseline run on NC3 allocation with reduced compute budget.

Goal: get preliminary numbers comparable to CEI in Table III. If PPO converges
to anything sensible, those numbers strengthen the Reviewer #4 response. If it
fails to converge, fall back to citation-based comparison with Davis/Robbins/
Lunday (ADP, 7.74% optimality gap).
"""

from __future__ import annotations
import json
import time
import sys
import os

# Suppress SB3 logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, '/home/claude/paper1_day1')
from rl_baseline_nc3 import NC3Config, NC3AllocationEnv


class NC3Gym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, scenario):
        super().__init__()
        self._env = NC3AllocationEnv(cfg, scenario)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._env.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=cfg.per_node_cap,
            shape=(self._env.action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)


def main():
    cfg = NC3Config()
    # Reduce budgets for compute-constrained run
    cfg.n_steps = 50            # halved from 100 for faster training
    cfg.train_timesteps = 30_000   # ~5-7 min instead of 30+
    cfg.n_envs = 4
    cfg.n_trials = 30           # eval trials per scenario

    print(f"PPO config: timesteps={cfg.train_timesteps}, n_envs={cfg.n_envs}, "
          f"steps_per_episode={cfg.n_steps}, eval_trials={cfg.n_trials}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    t0 = time.time()
    print("Building training envs (targeted scenario)...")
    env_fns = [lambda: NC3Gym(cfg, "targeted") for _ in range(cfg.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    print("Initializing PPO...")
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=512, batch_size=64,
        verbose=0, seed=cfg.seed,
    )
    print(f"Starting training ({cfg.train_timesteps} timesteps)...")
    model.learn(total_timesteps=cfg.train_timesteps, progress_bar=False)
    train_elapsed = time.time() - t0
    print(f"Training done in {train_elapsed:.1f}s")

    # Evaluate across three scenarios
    print("\nEvaluating across scenarios...")
    results = {}
    for scenario in ("random", "targeted", "cascade"):
        comm_rates = []
        gov_rates = []
        for trial in range(cfg.n_trials):
            env = NC3AllocationEnv(cfg, scenario)
            state, _ = env.reset(seed=cfg.rng_seed + trial)
            trial_comm = []
            trial_gov = []
            for step in range(cfg.n_steps):
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _, info = env.step(action)
                trial_comm.append(info['comm_success'])
                trial_gov.append(info['gov_compliance'])
                if done:
                    break
            comm_rates.append(float(np.mean(trial_comm)))
            gov_rates.append(float(np.mean(trial_gov)))
        results[scenario] = {
            'comm_success_mean': float(np.mean(comm_rates)),
            'comm_success_std': float(np.std(comm_rates)),
            'gov_compliance_mean': float(np.mean(gov_rates)),
            'gov_compliance_std': float(np.std(gov_rates)),
            'n_trials': cfg.n_trials,
        }
        print(f"  {scenario:>10s}: comm={results[scenario]['comm_success_mean']:.3f}, "
              f"gov={results[scenario]['gov_compliance_mean']:.3f}")

    eval_elapsed = time.time() - t0 - train_elapsed
    print(f"\nEvaluation done in {eval_elapsed:.1f}s")
    print(f"Total wall time: {time.time() - t0:.1f}s")

    out = {
        'training': {
            'timesteps': cfg.train_timesteps,
            'n_envs': cfg.n_envs,
            'wall_time_s': train_elapsed,
        },
        'evaluation': results,
        'eval_wall_time_s': eval_elapsed,
        'note': 'Reduced budget for compute constraints; full reproduction uses 200k timesteps, 500 trials per scenario',
    }
    with open('/home/claude/paper1_day1/ppo_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Results saved to ppo_results.json")


if __name__ == "__main__":
    main()
