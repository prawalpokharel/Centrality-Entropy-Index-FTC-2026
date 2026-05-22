"""
Option 1B: PPO baseline for sensor fusion CEI domain.
(FIXED: portable output paths)

USAGE:
    python rl_baseline_sensor_fusion_fixed.py

OUTPUT (saved next to script):
    sensor_fusion_ppo_results.json
    sensor_fusion_ppo_policy.zip

EXPECTED RUNTIME: ~1-2 min on Apple Silicon (24 nodes vs 12)
"""

import os
import numpy as np
import json
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_SENSING = 16
N_PROCESSING = 8
N_NODES = N_SENSING + N_PROCESSING
T_SLOTS = 500
W_WINDOW = 20
GOV_FLOOR = 0.60
TIER_G = np.array(
    [1.0] * 8 + [0.6] * 8 + [0.8] * 4 + [0.8] * 2 + [0.3] * 2
)
T1_MASK = np.zeros(N_NODES, dtype=bool)
T1_MASK[:8] = True

SCENARIOS = ['S1_nominal', 'S2_degradation', 'S3_jamming', 'S4_comms_out']


class SensorFusionCEIEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario='S1_nominal', seed=42):
        super().__init__()
        assert scenario in SCENARIOS
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_NODES * 2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_NODES,), dtype=np.float32
        )
        self.t = 0
        self.duty = np.full(N_NODES, 0.5, dtype=np.float32)

    def _quality_and_jamming(self):
        sq_base = np.ones(N_NODES, dtype=np.float32)
        jamming = np.zeros(N_NODES, dtype=np.float32)
        d_type = np.ones(N_NODES, dtype=np.float32)
        d_type[8:12] *= 0.85 + 0.15 * np.sin(2 * np.pi * self.t / 60)
        d_type[4:8] *= 0.9 + 0.1 * self.rng.random(4)

        if self.scenario == 'S2_degradation':
            if self.t < 120:
                d_type[0] *= max(0.3, 1.0 - self.t / 120.0)
            else:
                d_type[0] *= 0.3
        elif self.scenario == 'S3_jamming':
            if self.t >= 50:
                jamming[0] = 0.9
                jamming[4] = 0.8
                jamming[16] = 0.7
        elif self.scenario == 'S4_comms_out':
            if self.t >= 100:
                d_type *= 0.6

        sq = sq_base * d_type * (1.0 - jamming)
        return sq.astype(np.float32), jamming.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.duty = np.full(N_NODES, 0.5, dtype=np.float32)
        sq, jam = self._quality_and_jamming()
        return np.concatenate([sq, jam]).astype(np.float32), {}

    def step(self, action):
        self.duty = np.clip(action, 0.0, 1.0).astype(np.float32)
        sq, jam = self._quality_and_jamming()
        active_t1 = (sq[T1_MASK] > 0.3) & (self.duty[T1_MASK] > 0)
        availability = float(active_t1.mean())
        gov_compliance = float((self.duty[T1_MASK] >= GOV_FLOOR).mean())
        detect_quality = float(np.mean(sq[T1_MASK] * self.duty[T1_MASK]))
        reward = 0.6 * availability + 0.3 * gov_compliance + 0.1 * detect_quality
        self.t += 1
        terminated = self.t >= T_SLOTS
        truncated = False
        info = {
            "availability": availability,
            "governance_compliance": gov_compliance,
            "detect_quality": detect_quality,
            "mean_duty": float(self.duty.mean()),
        }
        return np.concatenate([sq, jam]).astype(np.float32), reward, terminated, truncated, info


def make_env(scenario, seed):
    def _init():
        env = SensorFusionCEIEnv(scenario=scenario, seed=seed)
        return Monitor(env)
    return _init


def evaluate(model, scenario, n_episodes=5, seed=2026):
    env = SensorFusionCEIEnv(scenario=scenario, seed=seed)
    all_avail, all_gov, all_dq, all_duty = [], [], [], []
    all_osc = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        prev_duty = env.duty.copy()
        ep_avail, ep_gov, ep_dq, ep_duty = [], [], [], []
        ep_osc = 0
        for _ in range(T_SLOTS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            ep_avail.append(info["availability"])
            ep_gov.append(info["governance_compliance"])
            ep_dq.append(info["detect_quality"])
            ep_duty.append(info["mean_duty"])
            d_change = np.abs(env.duty - prev_duty) > 0.1
            ep_osc += int(d_change.sum())
            prev_duty = env.duty.copy()
            if done:
                break
        all_avail.append(np.mean(ep_avail))
        all_gov.append(np.mean(ep_gov))
        all_dq.append(np.mean(ep_dq))
        all_duty.append(np.mean(ep_duty))
        all_osc.append(ep_osc)
    return {
        "availability_pct": round(100.0 * float(np.mean(all_avail)), 2),
        "governance_compliance_pct": round(100.0 * float(np.mean(all_gov)), 2),
        "detect_quality": round(float(np.mean(all_dq)), 3),
        "mean_duty": round(float(np.mean(all_duty)), 3),
        "total_oscillations_mean": round(float(np.mean(all_osc)), 1),
    }


def main():
    print("=" * 60)
    print("Option 1B: PPO Baseline for Sensor Fusion CEI Domain")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Nodes: {N_NODES}, Slots/episode: {T_SLOTS}")
    print()

    n_envs = 8
    vec_env = DummyVecEnv([make_env('S1_nominal', seed=42 + i) for i in range(n_envs)])
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

    policy_path = os.path.join(OUT_DIR, "sensor_fusion_ppo_policy")
    model.save(policy_path)
    print(f"\nPolicy saved to: {policy_path}.zip")

    all_results = {}
    print("\n" + "=" * 60)
    print("RESULTS BY SCENARIO (drop-in for Paper 2 Table 6 PPO column)")
    print("=" * 60)
    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        res = evaluate(model, scenario, n_episodes=5, seed=2026)
        all_results[scenario] = res
        for k, v in res.items():
            print(f"  {k:<32} : {v}")

    print("\n" + "=" * 60)
    print("CROSS-SCENARIO SUMMARY (Paper 2 Table 6 format)")
    print("=" * 60)
    print(f"  {'Metric':<26}{'Static':>9}{'Thresh':>8}{'CEI':>8}{'PPO':>9}")
    print(f"  {'Avail. (S1 Nominal)':<26}{'100%':>9}{'100%':>8}{'100%':>8}"
          f"{all_results['S1_nominal']['availability_pct']:>8}%")
    print(f"  {'Avail. (S3 Jamming)':<26}{'71.4%':>9}{'76.8%':>8}{'95.6%':>8}"
          f"{all_results['S3_jamming']['availability_pct']:>8}%")
    print(f"  {'Avail. (S4 COMMS)':<26}{'68.3%':>9}{'72.1%':>8}{'91.2%':>8}"
          f"{all_results['S4_comms_out']['availability_pct']:>8}%")
    print(f"  {'Detect. Qual. (S2)':<26}{'0.62':>9}{'0.68':>8}{'0.83':>8}"
          f"{all_results['S2_degradation']['detect_quality']:>9}")
    print(f"  {'Gov. Compliance (avg)':<26}{'82.1%':>9}{'78.4%':>8}{'97.3%':>8}"
          f"{round(np.mean([r['governance_compliance_pct'] for r in all_results.values()]),1):>8}%")

    results_path = os.path.join(OUT_DIR, "sensor_fusion_ppo_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON saved to: {results_path}")


if __name__ == "__main__":
    main()
