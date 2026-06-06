"""
Paper 1 Cloud Domain — AWS Health Dashboard Calibration

Calibrates the Paper 1 cloud-domain simulation against publicly reported
AWS Health Dashboard data from the March 1, 2026 Iranian drone strikes on
AWS Middle East infrastructure.

This script:
  1. Loads aws_incident_profile_march_2026.json
  2. Parameterizes a cloud-region simulation environment using the real
     91-service / 9-24-58 tier distribution from me-central-1
  3. Runs four allocators (Static, Reactive+Gov, CEI, PPO) plus a
     Lagrangian-PPO constrained-RL variant
  4. Outputs calibrated results JSON for Paper 1 cloud table integration

Usage:
    pip install gymnasium stable-baselines3 numpy
    python aws_incident_calibration.py [--quick]

Output:
    aws_calibration_results.json
    aws_calibration_summary.txt

HONESTY NOTE: This calibration parameterizes the simulation against the
publicly reported TOPOLOGY and TIMING of the event. It does not replay
proprietary telemetry. The framing in Paper 1 must reflect this.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


SCRIPT_DIR = Path(__file__).parent.absolute()
PROFILE_PATH = SCRIPT_DIR / "aws_incident_profile_march_2026.json"
OUT_RESULTS = SCRIPT_DIR / "aws_calibration_results.json"
OUT_SUMMARY = SCRIPT_DIR / "aws_calibration_summary.txt"


# ============================================================
# Load calibration profile
# ============================================================

with open(PROFILE_PATH, "r") as f:
    PROFILE = json.load(f)

ME_CENTRAL = PROFILE["regions_affected"]["me-central-1"]
ME_SOUTH   = PROFILE["regions_affected"]["me-south-1"]
CASCADE    = PROFILE["cascade_pattern"]
RECOVERY   = PROFILE["recovery_profile"]


# ============================================================
# Calibrated cloud env config (derived from AWS Health Dashboard data)
# ============================================================

@dataclass(frozen=True)
class CalibratedCloudConfig:
    # Topology calibrated to me-central-1
    n_services: int = ME_CENTRAL["services_impacted_total"]  # 91
    n_tier1: int    = ME_CENTRAL["tier1_disrupted_count"]    # 9
    n_tier2: int    = ME_CENTRAL["tier2_degraded_count"]     # 24
    n_tier3: int    = ME_CENTRAL["tier3_impacted_count"]     # 58
    n_azs: int      = 3
    # Timeline calibrated to incident (relative hours, scaled to slots)
    t_slots: int = 720    # 720 slots representing 7 days at 14-min resolution
    cascade_az_to_az_slot: int = 30       # intra-region cascade
    cascade_region_to_region_slot: int = 75  # cross-region propagation
    software_recovery_slot: int = 220       # partial software-mitigated
    physical_recovery_slot: int = 720       # full physical recovery
    # Allocator parameters (consistent with rest of Paper 1)
    gov_floor: float = 0.60
    theta_health: float = 0.40
    hysteresis_h: int = 18
    action_change_threshold: float = 0.10
    seed: int = 42


CFG = CalibratedCloudConfig()


# T1 services indices (criticality-1: EC2 core, EBS, S3, Lambda, Console, CLI)
T1_MASK = np.zeros(CFG.n_services, dtype=bool)
T1_MASK[: CFG.n_tier1] = True

T2_MASK = np.zeros(CFG.n_services, dtype=bool)
T2_MASK[CFG.n_tier1 : CFG.n_tier1 + CFG.n_tier2] = True

T3_MASK = np.zeros(CFG.n_services, dtype=bool)
T3_MASK[CFG.n_tier1 + CFG.n_tier2 :] = True

# Control-plane services are a subset of T1+T2 (IAM, CloudFormation analogs)
CONTROL_PLANE_MASK = np.zeros(CFG.n_services, dtype=bool)
CONTROL_PLANE_MASK[[0, 4, 9, 10]] = True  # Console, CLI, RDS, DynamoDB control APIs


# ============================================================
# Calibrated cloud environment
# ============================================================

class CalibratedCloudEnv(gym.Env):
    """
    Cloud allocation env calibrated against AWS me-central-1 March 2026 event.

    State: per-service health [0,1] (91 floats) + episode-phase scalar (1) = 92
    Action: per-service duty cycle [0,1] (91 floats)
    Reward: 0.45*availability + 0.30*gov_compliance + 0.15*throughput - 0.10*waste

    Adversarial timeline matches calibrated incident profile:
        slot 0:                kinetic strike on az2 (T1 services start failing)
        slot 30:               az3 cascade (intra-region propagation)
        slot 75:               cross-region cascade to me-south-1
        slot 220:              software-mitigated partial recovery (T1 services)
        slot 720:              physical recovery complete (T2, T3 services)
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, config: CalibratedCloudConfig = CFG):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(config.n_services + 1,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(config.n_services,),
            dtype=np.float32
        )
        self.t = 0
        self.duty = np.full(config.n_services, 0.5, dtype=np.float32)

    def _health(self):
        """Calibrated health per service over time."""
        cfg = self.config
        health = np.ones(cfg.n_services, dtype=np.float32)

        # Pre-strike: normal operations
        if self.t < 5:
            return health * (0.95 + 0.05 * self.rng.random(cfg.n_services)).astype(np.float32)

        # T+0: kinetic strike on az2 - T1 services lose ~1/3 (one AZ of three)
        if self.t >= 5:
            # T1 services in az2 fully impacted, az1+az3 carry load
            health[T1_MASK] *= 0.66  # 2/3 of AZs still functional

        # Control-plane fails first (cf. Medium analysis)
        if self.t >= 7:
            health[CONTROL_PLANE_MASK] *= 0.4

        # T+30 slots (~7 hours): mec1-az3 cascade
        if self.t >= cfg.cascade_az_to_az_slot:
            health[T1_MASK] *= 0.5   # 2/3 -> 1/3 (only one AZ left)
            health[T2_MASK] *= 0.65  # T2 services start degrading

        # T+75 slots (~17 hours): cross-region cascade to me-south-1
        if self.t >= cfg.cascade_region_to_region_slot:
            health[T2_MASK] *= 0.7   # additional degradation
            health[T3_MASK] *= 0.75  # T3 starts showing impact

        # T+220 slots (~52 hours): software-mitigated partial recovery for T1
        if self.t >= cfg.software_recovery_slot:
            health[T1_MASK] = np.minimum(
                health[T1_MASK] * 1.5 + 0.2,
                0.7
            )  # cap at 70% (still infrastructure-constrained)

        # T+720 slots (~7 days): full physical recovery
        if self.t >= cfg.physical_recovery_slot - 50:
            recovery_factor = min(1.0, (self.t - (cfg.physical_recovery_slot - 50)) / 50)
            health = health * (1 - recovery_factor) + recovery_factor * 0.95

        # Add small noise
        health = health + self.rng.normal(0, 0.03, size=cfg.n_services).astype(np.float32)
        return np.clip(health, 0.0, 1.0).astype(np.float32)

    def _obs(self, health):
        phase = np.array([self.t / max(1, self.config.t_slots)], dtype=np.float32)
        return np.concatenate([health, phase]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.duty = np.full(self.config.n_services, 0.5, dtype=np.float32)
        return self._obs(self._health()), {}

    def _metrics(self, health):
        cfg = self.config
        # T1 availability: tier-1 services with health > theta AND duty > floor
        active_t1 = (health[T1_MASK] >= cfg.theta_health) & (self.duty[T1_MASK] >= 0.05)
        # Require at least 5 of 9 T1 services healthy for region to count as "available"
        availability = float(active_t1.sum() >= 5)
        # Governance: T1 services must have duty >= 0.60
        gov_compliance = float((self.duty[T1_MASK] >= cfg.gov_floor).mean())
        # Throughput: weighted by health and duty
        throughput = float(np.mean(health * self.duty))
        # Waste: duty allocated to services with low health
        unhealthy = health < 0.3
        if unhealthy.any():
            waste = float(np.mean(self.duty[unhealthy]))
        else:
            waste = 0.0
        return {
            "availability": availability,
            "governance_compliance": gov_compliance,
            "throughput": throughput,
            "waste": waste,
            "mean_duty": float(self.duty.mean()),
        }

    def step(self, action):
        self.duty = np.clip(action, 0.0, 1.0).astype(np.float32)
        health = self._health()
        m = self._metrics(health)
        reward = (
            0.45 * m["availability"]
            + 0.30 * m["governance_compliance"]
            + 0.15 * m["throughput"]
            - 0.10 * m["waste"]
        )
        self.t += 1
        terminated = self.t >= self.config.t_slots
        return self._obs(health), float(reward), terminated, False, m


# ============================================================
# Lagrangian wrapper (consistent with Paper 2)
# ============================================================

GOV_TARGET = 0.95


class LagrangianWrapper(gym.Wrapper):
    def __init__(self, env, target=GOV_TARGET, initial_lambda=1.0):
        super().__init__(env)
        self.target = target
        self.lam = float(initial_lambda)

    def set_lambda(self, value):
        self.lam = max(0.0, float(value))

    def get_lambda(self):
        return float(self.lam)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        gov = float(info.get("governance_compliance", 0.0))
        violation = max(0.0, self.target - gov)
        adjusted = r - self.lam * violation
        info = dict(info)
        info.update({"raw_reward": r, "violation": violation, "lambda": self.lam})
        return obs, float(adjusted), terminated, truncated, info


class LambdaUpdateCallback(BaseCallback):
    def __init__(self, lambda_lr, target=GOV_TARGET, verbose=0):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.target = target
        self.violations = []
        self.lambda_history = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "violation" in info:
                self.violations.append(float(info["violation"]))
        return True

    def _on_rollout_end(self):
        if not self.violations:
            return
        mean_violation = float(np.mean(self.violations))
        current_lambda = float(self.training_env.env_method("get_lambda")[0])
        new_lambda = max(0.0, current_lambda + self.lambda_lr * mean_violation)
        self.training_env.env_method("set_lambda", new_lambda)
        self.lambda_history.append(new_lambda)
        self.violations.clear()


# ============================================================
# Allocators (analytical, non-RL)
# ============================================================

def run_static_allocator(seed: int, episodes: int) -> dict:
    """Static allocation: all services at duty 0.60."""
    rows = []
    for ep in range(episodes):
        env = CalibratedCloudEnv(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_metrics = []
        prev_duty = env.duty.copy()
        osc = 0
        for _ in range(CFG.t_slots):
            action = np.full(CFG.n_services, 0.60, dtype=np.float32)
            obs, _, done, _, info = env.step(action)
            ep_metrics.append(info)
            osc += int((np.abs(env.duty - prev_duty) > CFG.action_change_threshold).sum())
            prev_duty = env.duty.copy()
            if done:
                break
        rows.append({
            "availability_pct": 100.0 * float(np.mean([m["availability"] for m in ep_metrics])),
            "governance_compliance_pct": 100.0 * float(np.mean([m["governance_compliance"] for m in ep_metrics])),
            "throughput": float(np.mean([m["throughput"] for m in ep_metrics])),
            "waste": float(np.mean([m["waste"] for m in ep_metrics])),
            "oscillations": osc,
            "mean_duty": float(np.mean([m["mean_duty"] for m in ep_metrics])),
        })
    return aggregate_episodes(rows)


def run_reactive_gov_allocator(seed: int, episodes: int) -> dict:
    """Reactive+Gov: reduce duty when health high, boost when low; enforce gov floor."""
    rows = []
    for ep in range(episodes):
        env = CalibratedCloudEnv(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_metrics = []
        prev_duty = env.duty.copy()
        osc = 0
        for _ in range(CFG.t_slots):
            health = obs[:CFG.n_services]
            action = np.where(health >= 0.6, 0.30, 0.85).astype(np.float32)
            # Enforce gov floor on T1
            action[T1_MASK] = np.maximum(action[T1_MASK], CFG.gov_floor)
            obs, _, done, _, info = env.step(action)
            ep_metrics.append(info)
            osc += int((np.abs(env.duty - prev_duty) > CFG.action_change_threshold).sum())
            prev_duty = env.duty.copy()
            if done:
                break
        rows.append({
            "availability_pct": 100.0 * float(np.mean([m["availability"] for m in ep_metrics])),
            "governance_compliance_pct": 100.0 * float(np.mean([m["governance_compliance"] for m in ep_metrics])),
            "throughput": float(np.mean([m["throughput"] for m in ep_metrics])),
            "waste": float(np.mean([m["waste"] for m in ep_metrics])),
            "oscillations": osc,
            "mean_duty": float(np.mean([m["mean_duty"] for m in ep_metrics])),
        })
    return aggregate_episodes(rows)


def run_cei_allocator(seed: int, episodes: int) -> dict:
    """CEI: hysteresis-gated, governance-floored, entropy-aware allocation."""
    rows = []
    for ep in range(episodes):
        env = CalibratedCloudEnv(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_metrics = []
        prev_duty = env.duty.copy()
        osc = 0
        h_counter = np.zeros(CFG.n_services, dtype=np.int32)
        # Centrality weights: T1 > T2 > T3 (critical services have higher priority)
        centrality = np.where(T1_MASK, 1.0, np.where(T2_MASK, 0.6, 0.3))
        for _ in range(CFG.t_slots):
            health = obs[:CFG.n_services]
            # Entropy proxy: variance over a window of recent health values
            # (simplified: just current health uncertainty)
            entropy = 1.0 - np.abs(health - 0.5) * 2  # high when health near 0.5
            # CEI score per service
            cei_score = 0.5 * centrality + 0.3 * (1.0 - health) + 0.2 * entropy
            # Target duty: scale by CEI score
            target = np.clip(0.40 + 0.45 * cei_score, 0.15, 0.95).astype(np.float32)
            # Apply governance floor on T1
            target[T1_MASK] = np.maximum(target[T1_MASK], CFG.gov_floor)
            # Hysteresis gate
            allowed = (np.abs(target - env.duty) > CFG.action_change_threshold) & (h_counter == 0)
            action = np.where(allowed, target, env.duty)
            h_counter = np.where(allowed, CFG.hysteresis_h, np.maximum(0, h_counter - 1))
            obs, _, done, _, info = env.step(action)
            ep_metrics.append(info)
            osc += int((np.abs(env.duty - prev_duty) > CFG.action_change_threshold).sum())
            prev_duty = env.duty.copy()
            if done:
                break
        rows.append({
            "availability_pct": 100.0 * float(np.mean([m["availability"] for m in ep_metrics])),
            "governance_compliance_pct": 100.0 * float(np.mean([m["governance_compliance"] for m in ep_metrics])),
            "throughput": float(np.mean([m["throughput"] for m in ep_metrics])),
            "waste": float(np.mean([m["waste"] for m in ep_metrics])),
            "oscillations": osc,
            "mean_duty": float(np.mean([m["mean_duty"] for m in ep_metrics])),
        })
    return aggregate_episodes(rows)


def aggregate_episodes(rows: list) -> dict:
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    return {
        "availability_pct": round(mean("availability_pct"), 2),
        "governance_compliance_pct": round(mean("governance_compliance_pct"), 2),
        "throughput": round(mean("throughput"), 4),
        "waste_pct": round(100.0 * mean("waste"), 2),
        "oscillations_per_episode": round(mean("oscillations"), 1),
        "mean_duty_cycle": round(mean("mean_duty"), 3),
        "episodes": len(rows),
    }


# ============================================================
# PPO baseline
# ============================================================

def make_calibrated_env(seed):
    def _init():
        env = CalibratedCloudEnv(seed=seed)
        return Monitor(env)
    return _init


def make_lagrangian_env(seed):
    def _init():
        base = CalibratedCloudEnv(seed=seed)
        wrapped = LagrangianWrapper(base)
        return Monitor(wrapped)
    return _init


def evaluate_rl_model(model, episodes, lagrangian=False):
    rows = []
    for ep in range(episodes):
        if lagrangian:
            env = LagrangianWrapper(CalibratedCloudEnv(seed=2026 + ep))
        else:
            env = CalibratedCloudEnv(seed=2026 + ep)
        obs, _ = env.reset(seed=2026 + ep)
        base_env = env.env if lagrangian else env
        prev_duty = base_env.duty.copy()
        ep_metrics = []
        osc = 0
        for _ in range(CFG.t_slots):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            ep_metrics.append(info)
            osc += int((np.abs(base_env.duty - prev_duty) > CFG.action_change_threshold).sum())
            prev_duty = base_env.duty.copy()
            if done:
                break
        rows.append({
            "availability_pct": 100.0 * float(np.mean([m["availability"] for m in ep_metrics])),
            "governance_compliance_pct": 100.0 * float(np.mean([m["governance_compliance"] for m in ep_metrics])),
            "throughput": float(np.mean([m["throughput"] for m in ep_metrics])),
            "waste": float(np.mean([m["waste"] for m in ep_metrics])),
            "oscillations": osc,
            "mean_duty": float(np.mean([m["mean_duty"] for m in ep_metrics])),
        })
    return aggregate_episodes(rows)


def run_ppo_baseline(seed: int, episodes: int, timesteps: int) -> dict:
    vec_env = DummyVecEnv([make_calibrated_env(seed + i) for i in range(8)])
    model = PPO(
        "MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0, seed=seed
    )
    print(f"    [PPO] training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_seconds = time.time() - start
    print(f"    [PPO] training done in {train_seconds:.1f}s")
    model.save(str(SCRIPT_DIR / "cloud_ppo_policy"))
    res = evaluate_rl_model(model, episodes)
    res["training_seconds"] = round(train_seconds, 1)
    res["timesteps"] = timesteps
    return res


def run_lagrangian_ppo_baseline(seed: int, episodes: int, timesteps: int) -> dict:
    vec_env = DummyVecEnv([make_lagrangian_env(seed + i) for i in range(8)])
    model = PPO(
        "MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0, seed=seed
    )
    cb = LambdaUpdateCallback(lambda_lr=3e-3)
    print(f"    [Lagrangian-PPO] training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    train_seconds = time.time() - start
    print(f"    [Lagrangian-PPO] training done in {train_seconds:.1f}s")
    model.save(str(SCRIPT_DIR / "cloud_lagrangian_ppo_policy"))
    res = evaluate_rl_model(model, episodes, lagrangian=True)
    res["training_seconds"] = round(train_seconds, 1)
    res["timesteps"] = timesteps
    res["lambda_final"] = round(cb.lambda_history[-1], 4) if cb.lambda_history else 0.0
    return res


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 25k timesteps, 5 episodes (for verification)")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 25_000
        args.episodes = 5

    print("=" * 70)
    print("AWS Health Dashboard Calibrated Cloud Simulation")
    print("=" * 70)
    print(f"Incident: {PROFILE['incident_name']}")
    print(f"Start: {PROFILE['event_start_pst']}")
    print(f"Topology: {ME_CENTRAL['services_impacted_total']} services in me-central-1")
    print(f"Tier distribution: {ME_CENTRAL['tier1_disrupted_count']}/{ME_CENTRAL['tier2_degraded_count']}/{ME_CENTRAL['tier3_impacted_count']} (T1/T2/T3)")
    print(f"Cascade timing: {CASCADE['intra_region_propagation_hours']}h intra-region, {CASCADE['cross_region_propagation_hours']}h cross-region")
    print()
    print(f"Running {args.episodes} episodes per allocator (T_slots = {CFG.t_slots})")
    print()

    results = {
        "calibration_profile": {
            "incident_id": PROFILE["incident_id"],
            "incident_name": PROFILE["incident_name"],
            "event_start_utc": PROFILE["event_start_utc"],
            "topology": {
                "n_services": CFG.n_services,
                "tier1": CFG.n_tier1,
                "tier2": CFG.n_tier2,
                "tier3": CFG.n_tier3,
                "n_azs": CFG.n_azs,
            },
            "timeline": {
                "intra_region_cascade_hours": CASCADE["intra_region_propagation_hours"],
                "cross_region_cascade_hours": CASCADE["cross_region_propagation_hours"],
                "software_recovery_hours": RECOVERY["software_mitigation_partial_recovery_hours"],
                "physical_recovery_hours_estimate": RECOVERY["physical_recovery_hours_estimate"],
            },
        },
        "simulation_config": asdict(CFG),
        "allocator_results": {},
    }

    print("[1/5] Static allocator...")
    results["allocator_results"]["static"] = run_static_allocator(args.seed, args.episodes)
    print(f"      {results['allocator_results']['static']}")

    print("[2/5] Reactive+Gov allocator...")
    results["allocator_results"]["reactive_gov"] = run_reactive_gov_allocator(args.seed, args.episodes)
    print(f"      {results['allocator_results']['reactive_gov']}")

    print("[3/5] CEI allocator...")
    results["allocator_results"]["cei"] = run_cei_allocator(args.seed, args.episodes)
    print(f"      {results['allocator_results']['cei']}")

    print("[4/5] PPO baseline...")
    results["allocator_results"]["ppo"] = run_ppo_baseline(args.seed, args.episodes, args.timesteps)
    print(f"      {results['allocator_results']['ppo']}")

    print("[5/5] Lagrangian-PPO baseline...")
    results["allocator_results"]["lagrangian_ppo"] = run_lagrangian_ppo_baseline(args.seed, args.episodes, args.timesteps)
    print(f"      {results['allocator_results']['lagrangian_ppo']}")

    results["honesty_note"] = PROFILE["calibration_use_for_paper1"]["honesty_note"]

    # Save results
    with open(OUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUT_RESULTS}")

    # Pretty summary
    with open(OUT_SUMMARY, "w") as f:
        f.write("AWS Health Dashboard Calibrated Cloud Simulation - Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Incident: {PROFILE['incident_name']}\n")
        f.write(f"Topology: {CFG.n_services} services, {CFG.n_tier1}/{CFG.n_tier2}/{CFG.n_tier3} tier distribution\n\n")
        f.write(f"{'Allocator':<18}{'Avail %':>10}{'Gov %':>10}{'Tput':>10}{'Waste %':>10}{'Osc/ep':>10}\n")
        f.write("-" * 70 + "\n")
        for name, r in results["allocator_results"].items():
            f.write(f"{name:<18}{r['availability_pct']:>10.1f}{r['governance_compliance_pct']:>10.1f}"
                    f"{r['throughput']:>10.4f}{r['waste_pct']:>10.1f}{r['oscillations_per_episode']:>10.1f}\n")
    print(f"Summary saved: {OUT_SUMMARY}")

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL CALIBRATED RESULTS")
    print("=" * 70)
    print(f"{'Allocator':<18}{'Avail %':>10}{'Gov %':>10}{'Tput':>10}{'Waste %':>10}{'Osc/ep':>10}")
    print("-" * 70)
    for name, r in results["allocator_results"].items():
        print(f"{name:<18}{r['availability_pct']:>10.1f}{r['governance_compliance_pct']:>10.1f}"
              f"{r['throughput']:>10.4f}{r['waste_pct']:>10.1f}{r['oscillations_per_episode']:>10.1f}")


if __name__ == "__main__":
    main()
