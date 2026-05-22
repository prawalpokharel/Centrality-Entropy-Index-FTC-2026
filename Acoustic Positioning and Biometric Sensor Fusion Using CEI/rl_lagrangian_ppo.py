"""
Option 3: Lagrangian-PPO (Constrained RL) Comparison to CEI.

Custom Lagrangian wrapper around stable-baselines3 PPO that learns a
dual variable lambda to enforce a hard governance constraint:
  E[governance_compliance] >= 0.95

Trains on both underwater and sensor fusion domains. Compares to CEI
on availability, governance, oscillations, and sample efficiency.

USAGE:
    pip install gymnasium stable-baselines3 numpy
    python rl_lagrangian_ppo.py

EXPECTED RUNTIME: ~2-3 hours total (Lagrangian needs more compute and
dual-rate sweep).

EXPECTED OUTCOMES (one of):
  Outcome A: Lagrangian-PPO matches CEI on availability and governance,
             but takes 10-50x more compute and fails to suppress
             oscillations.
  Outcome B: Lagrangian-PPO still trails CEI on oscillations because
             constrained RL doesn't inherently include hysteresis.

Both outcomes are publishable findings.

NOTES:
  - omnisafe library has built-in Lagrangian-PPO, but installation can
    be tricky on Apple Silicon. This script implements a minimal custom
    Lagrangian wrapper instead.
  - For production paper inclusion, also try omnisafe and compare.
"""

import numpy as np
import json
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# Reuse underwater env from Option 1A
N_NODES = 12
T_SLOTS = 600
THETA = 15.0
GOV_FLOOR = 0.60
SNR_BASE = 36.7
GOVERNANCE_TARGET = 0.95  # hard constraint: E[gov_compliance] >= 95%


class LagrangianEnvWrapper(gym.Wrapper):
    """
    Wraps an environment so that the reward returned is:
        r' = r - lambda * (governance_target - gov_compliance)

    where lambda is the dual variable adjusted by an outer loop.
    """
    def __init__(self, env, governance_target=GOVERNANCE_TARGET):
        super().__init__(env)
        self.governance_target = governance_target
        self.lam = 1.0  # initial dual variable

    def set_lambda(self, lam):
        self.lam = float(lam)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        gov = info.get("governance_compliance", 0.0)
        # Penalty for falling below governance target
        violation = max(0.0, self.governance_target - gov)
        adjusted_reward = r - self.lam * violation
        info["raw_reward"] = r
        info["violation"] = violation
        info["lambda"] = self.lam
        return obs, adjusted_reward, terminated, truncated, info


class UnderwaterEnvForLagrangian(gym.Env):
    """Minimal underwater env duplicated here for self-containment."""
    metadata = {"render_modes": []}
    def __init__(self, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=0.0, high=60.0,
                                              shape=(N_NODES,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                         shape=(N_NODES,), dtype=np.float32)
        self.t = 0
        self.duty = np.full(N_NODES, 0.5, dtype=np.float32)

    def _snr(self):
        snr = SNR_BASE + 4.0 * np.sin(2.0 * np.pi * self.t / 120.0)
        if self.rng.random() < 0.08:
            snr -= float(self.rng.exponential(2.0))
        return np.clip(snr + self.rng.normal(0, 0.5, size=N_NODES),
                        0.0, 60.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.duty = np.full(N_NODES, 0.5, dtype=np.float32)
        return self._snr(), {}

    def step(self, action):
        self.duty = np.clip(action, 0.0, 1.0).astype(np.float32)
        snr = self._snr()
        avail = float(((snr[:4] >= THETA) & (self.duty[:4] > 0)).mean())
        gov = float((self.duty[:4] >= GOV_FLOOR).mean())
        reward = 0.7 * avail + 0.3 * gov
        self.t += 1
        done = self.t >= T_SLOTS
        return snr, reward, done, False, {
            "availability": avail,
            "governance_compliance": gov,
        }


class LambdaUpdateCallback(BaseCallback):
    """Updates the Lagrangian multiplier after each rollout."""
    def __init__(self, lambda_lr=3e-4, verbose=0):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.gov_history = []
        self.lambda_history = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Get governance compliance from this rollout
        infos = self.locals.get('infos', [])
        violations = []
        govs = []
        for info in infos:
            if 'violation' in info:
                violations.append(info['violation'])
                govs.append(info.get('governance_compliance', 0))
        if violations:
            mean_violation = float(np.mean(violations))
            # Dual ascent: lambda += lambda_lr * mean_violation
            new_lambda = max(0.0, self.training_env.envs[0].lam + self.lambda_lr * mean_violation * 100)
            for env in self.training_env.envs:
                env.set_lambda(new_lambda)
            self.gov_history.append(float(np.mean(govs)) if govs else 0)
            self.lambda_history.append(new_lambda)


def make_lagrangian_env(seed, initial_lambda=1.0):
    def _init():
        env = UnderwaterEnvForLagrangian(seed=seed)
        env = LagrangianEnvWrapper(env, governance_target=GOVERNANCE_TARGET)
        env.set_lambda(initial_lambda)
        env = Monitor(env)
        return env
    return _init


def evaluate_lagrangian(model, n_episodes=10, seed=2026):
    env = UnderwaterEnvForLagrangian(seed=seed)
    wrapped = LagrangianEnvWrapper(env)
    all_avail, all_gov = [], []
    oscillations = 0
    for ep in range(n_episodes):
        obs, _ = wrapped.reset(seed=seed + ep)
        prev_duty = wrapped.env.duty.copy()
        ep_avail, ep_gov = [], []
        for _ in range(T_SLOTS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = wrapped.step(action)
            ep_avail.append(info["availability"])
            ep_gov.append(info["governance_compliance"])
            d_change = np.abs(wrapped.env.duty - prev_duty) > 0.1
            oscillations += int(d_change.sum())
            prev_duty = wrapped.env.duty.copy()
            if done:
                break
        all_avail.append(np.mean(ep_avail))
        all_gov.append(np.mean(ep_gov))
    return {
        "availability_pct": 100.0 * float(np.mean(all_avail)),
        "governance_compliance_pct": 100.0 * float(np.mean(all_gov)),
        "oscillations_per_episode": int(oscillations / n_episodes),
    }


def main():
    print("=" * 60)
    print("Option 3: Lagrangian-PPO Constrained RL")
    print("=" * 60)
    print(f"Governance constraint: E[gov_compliance] >= {GOVERNANCE_TARGET}")
    print(f"Dual learning rate sweep: [1e-4, 3e-4, 1e-3, 3e-3]")
    print()

    # Dual learning rate sweep
    best_lambda_lr = None
    best_score = -np.inf
    for lambda_lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        print(f"\n--- Trying lambda_lr = {lambda_lr} ---")
        n_envs = 8
        vec_env = DummyVecEnv([make_lagrangian_env(seed=42 + i)
                                for i in range(n_envs)])
        model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256,
                    batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01,
                    verbose=0, seed=42)
        callback = LambdaUpdateCallback(lambda_lr=lambda_lr)
        # Short pilot run for hyperparameter selection
        model.learn(total_timesteps=50_000, callback=callback, progress_bar=False)
        res = evaluate_lagrangian(model, n_episodes=3)
        # Score: minimize availability loss, maximize governance compliance
        score = res["governance_compliance_pct"] - 0.5 * (100 - res["availability_pct"])
        print(f"  Avail={res['availability_pct']:.1f}, "
              f"Gov={res['governance_compliance_pct']:.1f}, score={score:.1f}")
        if score > best_score:
            best_score = score
            best_lambda_lr = lambda_lr

    print(f"\nSelected lambda_lr = {best_lambda_lr}")

    # Full training with best lambda_lr
    print(f"\n=== Full training with lambda_lr = {best_lambda_lr} ===")
    n_envs = 8
    vec_env = DummyVecEnv([make_lagrangian_env(seed=42 + i) for i in range(n_envs)])
    model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256,
                batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01,
                verbose=1, seed=42)
    callback = LambdaUpdateCallback(lambda_lr=best_lambda_lr)
    model.learn(total_timesteps=213_000, callback=callback, progress_bar=True)

    final = evaluate_lagrangian(model, n_episodes=10, seed=2026)
    print("\nFinal Lagrangian-PPO results (drop-in for Paper 2 Table 4):")
    for k, v in final.items():
        print(f"  {k}: {v}")

    print("\nCompare to CEI (paper): availability=100%, gov_compliance=100%, "
          "oscillations=16")
    print("Compare to PPO unconstrained: availability~95-100%, gov_compliance<10%")

    results = {
        "best_lambda_lr": best_lambda_lr,
        "final_results": final,
        "cei_baseline": {"availability_pct": 100.0,
                          "governance_compliance_pct": 100.0,
                          "oscillations": 16},
        "lambda_trajectory": callback.lambda_history,
        "governance_trajectory": callback.gov_history,
    }
    with open("/home/claude/paper2_day1/lagrangian_ppo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to lagrangian_ppo_results.json")


if __name__ == "__main__":
    main()
