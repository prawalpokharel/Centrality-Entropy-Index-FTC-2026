"""
Option 1A: PPO baseline for underwater acoustic CEI domain.
(FIXED: portable output paths — saves alongside the script)

Trains an unconstrained PPO agent on the same 12-node underwater
allocation problem evaluated in Paper 2 Section 6.

USAGE:
    pip install gymnasium stable-baselines3 numpy
    python rl_baseline_underwater_fixed.py

OUTPUT (saved next to the script):
    underwater_ppo_results.json  - Paper 2 Table 4 row
    underwater_ppo_policy.zip    - trained policy

OBSERVED ACTUAL RUNTIME: ~35 seconds on Apple Silicon
"""

import os
import numpy as np
import json
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Portable output directory: directory containing this script
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# PAPER 2 SECTION 6 PARAMETERS (matched exactly)
# ============================================================
N_NODES = 12
N_TRANSPONDERS = 4
N_RELAYS = 4
N_VEHICLES = 4
T_SLOTS = 600
W_WINDOW = 30
THETA = 15.0
GOV_FLOOR = 0.60
SNR_BASE = 36.7
MAX_DISTURBANCE = 8.0

TIER_G = np.array(
    [1.0] * N_TRANSPONDERS
    + [0.8] * N_RELAYS
    + [0.3] * N_VEHICLES
)
T1_MASK = np.zeros(N_NODES, dtype=bool)
T1_MASK[:N_TRANSPONDERS] = True


class UnderwaterCEIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0.0, high=60.0, shape=(N_NODES,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_NODES,), dtype=np.float32
        )
        self.t = 0
        self.duty = np.full(N_NODES, 0.5, dtype=np.float32)

    def _snr(self):
        snr = SNR_BASE + 4.0 * np.sin(2.0 * np.pi * self.t / 120.0)
        if self.rng.random() < 0.08:
            snr -= float(self.rng.exponential(2.0))
        return np.clip(
            snr + self.rng.normal(0, 0.5, size=N_NODES),
            0.0, 60.0
        ).astype(np.float32)

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
        active_t1 = (snr[:N_TRANSPONDERS] >= THETA) & (self.duty[:N_TRANSPONDERS] > 0)
        availability = float(active_t1.mean())
        gov_compliance = float((self.duty[T1_MASK] >= GOV_FLOOR).mean())
        reward = 0.7 * availability + 0.3 * gov_compliance
        self.t += 1
        terminated = self.t >= T_SLOTS
        truncated = False
        info = {
            "availability": availability,
            "governance_compliance": gov_compliance,
            "mean_duty": float(self.duty.mean()),
            "mean_t1_duty": float(self.duty[T1_MASK].mean()),
        }
        return snr, reward, terminated, truncated, info


def make_env(seed):
    def _init():
        env = UnderwaterCEIEnv(seed=seed)
        env = Monitor(env)
        return env
    return _init


def evaluate(model, n_episodes=10, seed=2026):
    env = UnderwaterCEIEnv(seed=seed)
    all_avail = []
    all_gov = []
    all_duty = []
    all_t1_duty = []
    all_oscillations = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        prev_duty = env.duty.copy()
        ep_avail, ep_gov, ep_duty, ep_t1_duty = [], [], [], []
        ep_oscillations = 0
        for _ in range(T_SLOTS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            ep_avail.append(info["availability"])
            ep_gov.append(info["governance_compliance"])
            ep_duty.append(info["mean_duty"])
            ep_t1_duty.append(info["mean_t1_duty"])
            d_change = np.abs(env.duty - prev_duty) > 0.1
            ep_oscillations += int(d_change.sum())
            prev_duty = env.duty.copy()
            if done:
                break
        all_avail.append(np.mean(ep_avail))
        all_gov.append(np.mean(ep_gov))
        all_duty.append(np.mean(ep_duty))
        all_t1_duty.append(np.mean(ep_t1_duty))
        all_oscillations.append(ep_oscillations)
    return {
        "positioning_availability_pct": round(100.0 * float(np.mean(all_avail)), 2),
        "governance_compliance_pct": round(100.0 * float(np.mean(all_gov)), 2),
        "mean_duty_cycle": round(float(np.mean(all_duty)), 3),
        "mean_t1_duty_cycle": round(float(np.mean(all_t1_duty)), 3),
        "total_oscillations_mean": round(float(np.mean(all_oscillations)), 1),
        "total_oscillations_std": round(float(np.std(all_oscillations)), 1),
        "n_episodes": n_episodes,
    }


def main():
    print("=" * 60)
    print("Option 1A: PPO Baseline for Underwater Acoustic CEI Domain")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Nodes: {N_NODES}, Slots/episode: {T_SLOTS}")
    print(f"Reward: 0.7 * availability + 0.3 * governance_compliance")
    print()

    n_envs = 8
    vec_env = DummyVecEnv([make_env(seed=42 + i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=42,
    )

    total_timesteps = 213_000
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    policy_path = os.path.join(OUT_DIR, "underwater_ppo_policy")
    model.save(policy_path)
    print(f"\nPolicy saved to: {policy_path}.zip")

    print("\nEvaluating trained policy across 10 episodes (seed=2026)...")
    results = evaluate(model, n_episodes=10, seed=2026)

    print("\n" + "=" * 60)
    print("RESULTS (drop-in for Paper 2 Table 4 PPO column)")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<35} : {v}")

    print("\nComparison vs Paper 2 baselines:")
    print(f"  {'Metric':<32}{'Static':>8}{'R+Gov':>8}{'CEI':>8}{'PPO':>10}")
    print(f"  {'Positioning Availability (%)':<32}{'100':>8}{'100':>8}{'100':>8}"
          f"{results['positioning_availability_pct']:>10}")
    print(f"  {'Total Oscillations':<32}{'0':>8}{'32':>8}{'16':>8}"
          f"{int(results['total_oscillations_mean']):>10}")
    print(f"  {'Mean Duty Cycle':<32}{'0.600':>8}{'0.284':>8}{'0.455':>8}"
          f"{results['mean_duty_cycle']:>10}")
    print(f"  {'Gov. Compliance (%)':<32}{'-':>8}{'100':>8}{'100':>8}"
          f"{results['governance_compliance_pct']:>10}")

    results_path = os.path.join(OUT_DIR, "underwater_ppo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON saved to: {results_path}")


if __name__ == "__main__":
    main()
