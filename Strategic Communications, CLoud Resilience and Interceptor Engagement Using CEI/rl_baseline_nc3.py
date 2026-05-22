"""
RL Baseline for NC3 Resource Allocation
=========================================
Reinforcement learning allocator for direct comparison against CEI in the
NC3 strategic communications domain (Paper 1, Section IV).

Addresses Reviewer #4's request for a stronger learning-based baseline beyond
the existing static, uniform, centrality-only, MDP, and robust-optimization
baselines reported in Table III.

Design choices:
- Topology, scenarios, trials, and metrics match the CEI evaluation exactly so
  results plug directly into Table III without redefinition.
- Continuous-action PPO is used because per-node allocation is continuous and
  the action space scales linearly with |V|.
- The harness is structured so the trained policy can be evaluated under all
  three scenarios (random / targeted / cascade) with the same seed protocol.

Installation:
    pip install --break-system-packages gymnasium stable-baselines3[extra] \\
        networkx numpy

Usage:
    python rl_baseline_nc3.py --train          # train PPO on targeted scenario
    python rl_baseline_nc3.py --evaluate       # 500-trial evaluation, all scenarios
    python rl_baseline_nc3.py --quick-check    # smoke test that env works
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx


# =============================================================================
# Configuration (matches CEI NC3 evaluation in paper Section IV.B)
# =============================================================================

@dataclass
class NC3Config:
    # Topology (paper §IV.B)
    n_backbone: int = 40              # T1+T2: Barabasi-Albert preferential attachment
    n_ground: int = 160               # T3+T4: Watts-Strogatz small-world
    ba_m: int = 3                     # BA preferential-attachment parameter
    ws_k: int = 4                     # WS small-world neighbors
    ws_beta: float = 0.3              # WS rewiring probability
    target_edges: int = 1600          # approximate total edges per paper

    # Episode / evaluation (paper §IV.B, §IV.D)
    n_steps: int = 100                # time steps per Monte Carlo trial
    n_trials: int = 500               # Monte Carlo trials per scenario
    total_budget: float = 100.0       # resource pool

    # Tier proportions (paper §III.D)
    tier_t1_frac: float = 0.05
    tier_t2_frac: float = 0.15
    tier_t3_frac: float = 0.40
    tier_t4_frac: float = 0.40

    # Tier-specific governance floors (paper §III.E)
    gov_floor_t1: float = 0.8
    gov_floor_t2: float = 0.5
    gov_floor_t3: float = 0.0
    gov_floor_t4: float = 0.0

    # Per-node allocation cap (prevents trivial all-to-one solutions)
    per_node_cap: float = 5.0

    # Scenario parameters (paper §IV.B)
    random_failure_prob: float = 0.10
    targeted_attack_k: int = 5         # top-k centrality nodes disabled
    cascade_lambda: float = 2.0

    # RL training
    train_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_envs: int = 8

    # Reproducibility
    seed: int = 42
    np_seed: int = 42                  # matches CEI harness (seeds: np=42, rng=2026)
    rng_seed: int = 2026


# =============================================================================
# Topology generation (paper §IV.B)
# =============================================================================

def build_nc3_topology(cfg: NC3Config) -> nx.DiGraph:
    """
    200-node hybrid graph: BA backbone (n=40, m=3) + WS ground (n=160, beta=0.3,
    k=4). Cross-layer links connect ground nodes to backbone by proximity.
    Tiers assigned: top-betweenness backbone -> T1, next -> T2, ground split
    T3/T4 randomly. Edge reliabilities ~ Beta(8, 2).
    """
    rng = np.random.default_rng(cfg.np_seed)

    backbone = nx.barabasi_albert_graph(cfg.n_backbone, cfg.ba_m, seed=cfg.np_seed)
    ground = nx.watts_strogatz_graph(
        cfg.n_ground, cfg.ws_k, cfg.ws_beta, seed=cfg.np_seed + 1
    )

    # Merge into single directed graph (relabel ground)
    G = nx.DiGraph()
    G.add_nodes_from(backbone.nodes())
    for u, v in backbone.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    relabel = {n: n + cfg.n_backbone for n in ground.nodes()}
    ground = nx.relabel_nodes(ground, relabel)
    G.add_nodes_from(ground.nodes())
    for u, v in ground.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    # Cross-layer links: each ground node connects to 2 backbone nodes
    backbone_nodes = list(range(cfg.n_backbone))
    for g in ground.nodes():
        targets = rng.choice(backbone_nodes, size=2, replace=False)
        for t in targets:
            G.add_edge(g, int(t))
            G.add_edge(int(t), g)

    # Assign tiers (top-betweenness in backbone are T1)
    bb_sub = G.subgraph(backbone_nodes)
    bb_cent = nx.betweenness_centrality(bb_sub.to_undirected())
    bb_sorted = sorted(bb_cent, key=bb_cent.get, reverse=True)

    n_total = cfg.n_backbone + cfg.n_ground
    n_t1 = max(1, int(n_total * cfg.tier_t1_frac))
    n_t2 = max(1, int(n_total * cfg.tier_t2_frac))
    n_t3 = int(n_total * cfg.tier_t3_frac)

    tier_map: Dict[int, int] = {}
    for n in bb_sorted[:n_t1]:
        tier_map[n] = 1
    for n in bb_sorted[n_t1:n_t1 + n_t2]:
        tier_map[n] = 2
    for n in bb_sorted[n_t1 + n_t2:]:
        tier_map[n] = 2  # remaining backbone -> T2

    ground_list = list(ground.nodes())
    rng.shuffle(ground_list)
    for n in ground_list[:n_t3]:
        tier_map[n] = 3
    for n in ground_list[n_t3:]:
        tier_map[n] = 4

    for n in G.nodes():
        tier_map.setdefault(n, 3)
    nx.set_node_attributes(G, tier_map, 'tier')

    # Edge reliabilities ~ Beta(8, 2)
    for u, v in G.edges():
        G.edges[u, v]['reliability'] = float(rng.beta(8, 2))

    return G


# =============================================================================
# NC3 allocation environment (Gymnasium-compatible)
# =============================================================================

class NC3AllocationEnv:
    """
    State: per-node [status, mean reliability, tier_critical_flag, time_frac]
    Action: per-node allocation in [0, per_node_cap], rescaled to budget
    Reward: 0.7 * comm_success + 0.3 * gov_compliance

    Comm success = fraction of T1 nodes connected (via working subgraph) to
    at least one other T1 or T2 node. Governance compliance = fraction of T1
    and T2 nodes receiving allocation >= tier floor.
    """

    def __init__(self, cfg: NC3Config, scenario: str = "targeted"):
        assert scenario in ("random", "targeted", "cascade")
        self.cfg = cfg
        self.scenario = scenario
        self.graph = build_nc3_topology(cfg)
        self.node_list = list(self.graph.nodes())
        self.n_nodes = len(self.node_list)
        self.tiers = np.array(
            [self.graph.nodes[n]['tier'] for n in self.node_list], dtype=int
        )
        self.rng = np.random.default_rng(cfg.rng_seed)

        # Precompute centrality ranking (for targeted attack)
        und = self.graph.to_undirected()
        self.cent = nx.betweenness_centrality(und)
        self.cent_sorted = sorted(self.cent, key=self.cent.get, reverse=True)

        self.state_dim = self.n_nodes * 4
        self.action_dim = self.n_nodes

        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.disabled = np.zeros(self.n_nodes, dtype=bool)
        self.status = np.zeros(self.n_nodes, dtype=int)  # 0=ok, 1=deg, 2=comp, 3=off
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, n in enumerate(self.node_list):
            edges = list(self.graph.edges(n))
            mean_rel = (
                np.mean([self.graph.edges[u, v]['reliability'] for u, v in edges])
                if edges else 0.0
            )
            tier = self.tiers[i]
            state[i * 4 + 0] = self.status[i] / 3.0
            state[i * 4 + 1] = mean_rel
            state[i * 4 + 2] = (5 - tier) / 4.0
            state[i * 4 + 3] = self.t / self.cfg.n_steps
        return state

    def _apply_disruption(self):
        if self.scenario == "random":
            for i in range(self.n_nodes):
                if not self.disabled[i] and self.rng.random() < self.cfg.random_failure_prob:
                    self.disabled[i] = True
                    self.status[i] = 3
        elif self.scenario == "targeted":
            if self.t < self.cfg.targeted_attack_k:
                # Disable next top-centrality node not yet disabled
                for node in self.cent_sorted:
                    idx = self.node_list.index(node)
                    if not self.disabled[idx]:
                        self.disabled[idx] = True
                        self.status[idx] = 3
                        break
        elif self.scenario == "cascade":
            new_failures = []
            for i, n in enumerate(self.node_list):
                if self.disabled[i]:
                    for nbr in self.graph.neighbors(n):
                        nbr_idx = self.node_list.index(nbr)
                        if not self.disabled[nbr_idx]:
                            p = self.cfg.random_failure_prob * self.cfg.cascade_lambda
                            if self.rng.random() < p:
                                new_failures.append(nbr_idx)
            for i in new_failures:
                self.disabled[i] = True
                self.status[i] = 3

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, 0, self.cfg.per_node_cap)
        s = a.sum()
        if s > self.cfg.total_budget and s > 0:
            a = a / s * self.cfg.total_budget
        return a

    def _comm_success(self) -> float:
        working_nodes = [self.node_list[i] for i in range(self.n_nodes) if not self.disabled[i]]
        if len(working_nodes) < 2:
            return 0.0
        sub = self.graph.subgraph(working_nodes).to_undirected()

        critical = [
            self.node_list[i] for i in range(self.n_nodes)
            if self.tiers[i] in (1, 2) and not self.disabled[i]
        ]
        if len(critical) < 2:
            return 0.0
        connected = 0
        for n in critical:
            for m in critical:
                if n == m:
                    continue
                if sub.has_node(n) and sub.has_node(m):
                    try:
                        if nx.has_path(sub, n, m):
                            connected += 1
                            break
                    except nx.NetworkXError:
                        pass
        return connected / max(1, len(critical))

    def _gov_compliance(self, action: np.ndarray) -> float:
        floors = {1: self.cfg.gov_floor_t1, 2: self.cfg.gov_floor_t2,
                  3: self.cfg.gov_floor_t3, 4: self.cfg.gov_floor_t4}
        critical_idx = [i for i in range(self.n_nodes) if self.tiers[i] in (1, 2)]
        if not critical_idx:
            return 1.0
        compliant = sum(1 for i in critical_idx if action[i] >= floors[self.tiers[i]])
        return compliant / len(critical_idx)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = self._normalize_action(np.asarray(action, dtype=np.float32))
        self._apply_disruption()
        comm = self._comm_success()
        gov = self._gov_compliance(action)
        reward = 0.7 * comm + 0.3 * gov
        self.t += 1
        done = self.t >= self.cfg.n_steps
        info = {
            'comm_success': comm,
            'gov_compliance': gov,
            'n_disabled': int(self.disabled.sum()),
        }
        return self._get_state(), reward, done, False, info


# =============================================================================
# Gymnasium wrapper (only imported if gymnasium installed)
# =============================================================================

def make_gym_env(cfg: NC3Config, scenario: str):
    import gymnasium as gym
    from gymnasium import spaces

    class _Wrap(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
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

    return _Wrap()


# =============================================================================
# Training and evaluation
# =============================================================================

def train_ppo(cfg: NC3Config, scenario: str = "targeted", out_path: str = "ppo_nc3.zip"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_fns = [lambda: make_gym_env(cfg, scenario) for _ in range(cfg.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=cfg.train_timesteps)
    model.save(out_path)
    print(f"Saved trained model to {out_path}")
    return model


def evaluate_ppo(cfg: NC3Config, model_path: str = "ppo_nc3.zip") -> Dict:
    """500-trial Monte Carlo evaluation across all three scenarios."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    results: Dict[str, dict] = {}

    for scenario in ("random", "targeted", "cascade"):
        comm_rates: List[float] = []
        gov_rates: List[float] = []
        recovery_steps: List[int] = []

        for trial in range(cfg.n_trials):
            trial_cfg = NC3Config(**{**asdict(cfg), 'rng_seed': cfg.rng_seed + trial})
            env = NC3AllocationEnv(trial_cfg, scenario)
            state, _ = env.reset(seed=trial_cfg.rng_seed + trial)
            trial_comm: List[float] = []
            trial_gov: List[float] = []
            recovery_t = None
            for step in range(trial_cfg.n_steps):
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, _, info = env.step(action)
                trial_comm.append(info['comm_success'])
                trial_gov.append(info['gov_compliance'])
                if info['comm_success'] > 0.9 and recovery_t is None and step > cfg.targeted_attack_k:
                    recovery_t = step - cfg.targeted_attack_k
                if done:
                    break
            comm_rates.append(float(np.mean(trial_comm)))
            gov_rates.append(float(np.mean(trial_gov)))
            if recovery_t is not None:
                recovery_steps.append(recovery_t)

        results[scenario] = {
            'comm_success_mean': float(np.mean(comm_rates)),
            'comm_success_ci95': (
                float(np.percentile(comm_rates, 2.5)),
                float(np.percentile(comm_rates, 97.5)),
            ),
            'gov_compliance_mean': float(np.mean(gov_rates)),
            'recovery_steps_mean': float(np.mean(recovery_steps)) if recovery_steps else None,
            'n_trials': len(comm_rates),
        }

    return results


def quick_check(cfg: NC3Config):
    """Smoke test: build env, take random actions, print metrics."""
    print("=" * 60)
    print("Quick check: topology + environment")
    print("=" * 60)
    G = build_nc3_topology(cfg)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    tier_counts: Dict[int, int] = {}
    for n in G.nodes():
        t = G.nodes[n]['tier']
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Tier distribution: {dict(sorted(tier_counts.items()))}")

    for scenario in ("random", "targeted", "cascade"):
        env = NC3AllocationEnv(cfg, scenario)
        state, _ = env.reset(seed=cfg.rng_seed)
        rewards: List[float] = []
        for _ in range(20):
            action = np.random.dirichlet(np.ones(env.action_dim)) * cfg.total_budget
            state, r, done, _, info = env.step(action)
            rewards.append(r)
            if done:
                break
        print(f"{scenario:>10s}: 20-step mean reward = {np.mean(rewards):.3f}, "
              f"final_disabled = {info['n_disabled']}, "
              f"final_comm = {info['comm_success']:.3f}, "
              f"final_gov = {info['gov_compliance']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train PPO")
    parser.add_argument("--evaluate", action="store_true", help="500-trial evaluation")
    parser.add_argument("--quick-check", action="store_true", help="env smoke test")
    parser.add_argument("--model-path", type=str, default="ppo_nc3.zip")
    parser.add_argument("--out", type=str, default="rl_baseline_results.json")
    args = parser.parse_args()

    cfg = NC3Config()

    if args.quick_check:
        quick_check(cfg)
    if args.train:
        train_ppo(cfg, scenario="targeted", out_path=args.model_path)
    if args.evaluate:
        results = evaluate_ppo(cfg, model_path=args.model_path)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.out}")
        print(json.dumps(results, indent=2))

    if not any([args.quick_check, args.train, args.evaluate]):
        print("No action specified. Run with --quick-check first to verify env, "
              "then --train, then --evaluate.")
