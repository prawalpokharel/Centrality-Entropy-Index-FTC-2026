"""
Minimal CEI reference allocator for Paper 1 NC3 domain.

Implements Equations (1)-(5) and the allocation problem in Equation (4) from
Section III, on the 200-node NC3 topology defined in Section IV.B. Used as
the comparison baseline for the PPO RL allocator and as the substrate for
the telemetry degradation experiments.

Not a full re-implementation of the production CEI engine. Captures only what
is needed to reproduce paper-aligned metrics under nominal and degraded
telemetry conditions.
"""

from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx

from rl_baseline_nc3 import (
    NC3Config, build_nc3_topology, NC3AllocationEnv,
)


# =============================================================================
# Telemetry degradation models
# =============================================================================

class TelemetryChannel:
    """
    Wraps observation extraction to inject realistic degradation patterns
    matching what would be observed under adversarial conditions or
    constrained network operation.

    Three independent failure modes (paper Section VIII, new subsection):
      - Lag:     observations are L steps stale
      - Noise:   gaussian noise added to reliability + entropy inputs
      - Dropout: fraction of nodes report no telemetry this step
    """
    def __init__(self, lag: int = 0, noise: float = 0.0, dropout: float = 0.0,
                 seed: int = 0):
        self.lag = lag
        self.noise = noise
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.history: List[np.ndarray] = []

    def observe(self, reliability: np.ndarray, entropy: np.ndarray,
                disabled: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns degraded views of (reliability, entropy, observed_disabled).
        Disabled mask in particular: dropout means we don't know it's disabled.
        """
        rel = reliability.copy()
        ent = entropy.copy()
        # Lag: use historical observation
        if self.lag > 0:
            self.history.append((rel.copy(), ent.copy(), disabled.copy()))
            if len(self.history) > self.lag:
                rel, ent, dis = self.history[-self.lag - 1]
            else:
                dis = disabled
        else:
            dis = disabled
        # Noise: gaussian on reliability and entropy
        if self.noise > 0:
            rel = rel + self.rng.normal(0, self.noise, size=rel.shape)
            rel = np.clip(rel, 0, 1)
            ent = ent + self.rng.normal(0, self.noise, size=ent.shape)
            ent = np.clip(ent, 0, np.log2(4))
        # Dropout: a fraction of nodes report nothing (use last known reliability=0.5)
        if self.dropout > 0:
            mask = self.rng.random(rel.shape) < self.dropout
            rel[mask] = 0.5  # uninformative prior
            ent[mask] = 1.0  # max uncertainty
        return rel, ent, dis


# =============================================================================
# CEI Allocator (paper Equations (1)-(4))
# =============================================================================

class CEIAllocator:
    """
    Applies CEI scoring (Eq. 3) and concave-utility allocation (Eq. 4).
    Weights are fixed at paper defaults (0.4, 0.2, 0.4); the stability-loss
    weight adaptation (Eq. 5) is omitted in this minimal reference since the
    comparison runs are short enough that fixed weights are sufficient.
    """
    def __init__(self, cfg: NC3Config,
                 alpha: float = 0.4, beta: float = 0.2, gamma: float = 0.4):
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _shannon_entropy(reliability: np.ndarray) -> np.ndarray:
        """
        Discretized reliability into 4 bins -> Shannon entropy proxy per node.
        Single value implementation: per-node entropy of (reliability,
        1-reliability) approximation for a binary working/degraded state model.
        """
        p = np.clip(reliability, 1e-6, 1 - 1e-6)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def allocate(self, centrality: np.ndarray, reliability: np.ndarray,
                 entropy: np.ndarray, tiers: np.ndarray,
                 disabled: np.ndarray) -> np.ndarray:
        """
        Returns per-node allocation x respecting:
          - budget: sum(x) <= R
          - per-node cap: x_i <= cap
          - governance floors for T1 (0.8) and T2 (0.5)
          - concave utility u(x) = sqrt(x)
        Implementation: closed-form water-filling on CEI scores after
        governance floors are satisfied.
        """
        n = len(centrality)
        # Governance score per tier
        g_value = np.where(tiers == 1, 1.0,
                  np.where(tiers == 2, 0.8,
                  np.where(tiers == 3, 0.6, 0.3)))
        floors = np.where(tiers == 1, self.cfg.gov_floor_t1,
                 np.where(tiers == 2, self.cfg.gov_floor_t2, 0.0))
        # CEI score (Eq. 3)
        cei = (self.alpha * centrality +
               self.beta * entropy / max(entropy.max(), 1e-6) +
               self.gamma * g_value)
        # Disabled nodes get zero allocation
        cei = np.where(disabled, 0.0, cei)
        # Step 1: satisfy floors
        x = np.where(disabled, 0.0, floors)
        used = x.sum()
        remaining = max(0.0, self.cfg.total_budget - used)
        # Step 2: distribute remaining proportional to CEI (concave-utility
        # water-filling approximation)
        active_mask = (~disabled) & (cei > 0)
        if remaining > 0 and active_mask.any():
            scores = cei * active_mask
            weights = scores / scores.sum()
            extra = weights * remaining
            x = x + extra
        # Step 3: clamp to per-node cap
        x = np.clip(x, 0.0, self.cfg.per_node_cap)
        # Re-normalize if we exceeded budget after clamping (unlikely with
        # paper-sized parameters but kept for safety)
        if x.sum() > self.cfg.total_budget:
            x = x / x.sum() * self.cfg.total_budget
        return x


# =============================================================================
# Evaluation under nominal and degraded telemetry
# =============================================================================

def _per_step_extract(env: NC3AllocationEnv) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Pull current ground-truth telemetry from the environment."""
    n = env.n_nodes
    reliability = np.zeros(n)
    for i, node in enumerate(env.node_list):
        edges = list(env.graph.edges(node))
        if edges:
            reliability[i] = np.mean(
                [env.graph.edges[u, v]['reliability'] for u, v in edges]
            )
    # Centrality (paper Eq. 1, simplified to betweenness only for reference)
    cent_arr = np.array(
        [env.cent[env.node_list[i]] for i in range(n)], dtype=np.float32
    )
    cent_arr = cent_arr / max(cent_arr.max(), 1e-6)
    entropy = CEIAllocator._shannon_entropy(reliability)
    return cent_arr, reliability, entropy, env.tiers, env.disabled.copy()


def evaluate_cei(cfg: NC3Config, scenario: str,
                 telemetry: Optional[TelemetryChannel] = None,
                 n_trials: int = 100) -> Dict:
    """Evaluate the CEI reference allocator with optional telemetry degradation."""
    allocator = CEIAllocator(cfg)
    comm: List[float] = []
    gov: List[float] = []
    for trial in range(n_trials):
        trial_cfg = NC3Config(**{**asdict(cfg), 'rng_seed': cfg.rng_seed + trial})
        env = NC3AllocationEnv(trial_cfg, scenario)
        env.reset(seed=trial_cfg.rng_seed + trial)
        if telemetry is not None:
            telemetry.history.clear()
        trial_comm: List[float] = []
        trial_gov: List[float] = []
        for _ in range(trial_cfg.n_steps):
            cent_g, rel_g, ent_g, tiers_g, dis_g = _per_step_extract(env)
            if telemetry is not None:
                rel_obs, ent_obs, dis_obs = telemetry.observe(rel_g, ent_g, dis_g)
            else:
                rel_obs, ent_obs, dis_obs = rel_g, ent_g, dis_g
            action = allocator.allocate(cent_g, rel_obs, ent_obs, tiers_g, dis_obs)
            _, _, done, _, info = env.step(action)
            trial_comm.append(info['comm_success'])
            trial_gov.append(info['gov_compliance'])
            if done:
                break
        comm.append(float(np.mean(trial_comm)))
        gov.append(float(np.mean(trial_gov)))
    return {
        'comm_success_mean': float(np.mean(comm)),
        'comm_success_ci95': (
            float(np.percentile(comm, 2.5)),
            float(np.percentile(comm, 97.5)),
        ),
        'gov_compliance_mean': float(np.mean(gov)),
        'n_trials': len(comm),
    }


if __name__ == "__main__":
    cfg = NC3Config()
    print("CEI reference - nominal telemetry")
    print("=" * 60)
    for scenario in ("random", "targeted", "cascade"):
        r = evaluate_cei(cfg, scenario, telemetry=None, n_trials=50)
        print(f"  {scenario:>10s}: comm={r['comm_success_mean']:.3f} "
              f"[{r['comm_success_ci95'][0]:.3f}, {r['comm_success_ci95'][1]:.3f}], "
              f"gov={r['gov_compliance_mean']:.3f}")
