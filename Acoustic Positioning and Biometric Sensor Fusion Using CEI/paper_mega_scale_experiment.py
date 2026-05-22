"""
paper2_mega_scale_experiment.py
================================
Real measured scaling experiment for paper 2 (sensor network CEI).

We construct a synthetic heterogeneous sensor network at four scales:
N = 24, 64, 256, 1024 nodes.

At each scale we measure:
  - Total time to compute CEI scores for one slot
  - Convergence rate of the adaptive weight update
  - Steady-state governance compliance
  - Bandwidth waste vs reactive baseline

Topology: heterogeneous Watts-Strogatz small-world graph
(matches sensor network connectivity patterns -- short paths,
 high clustering, low long-distance link count).

This is intentionally simpler than paper3's experiment because
paper2's allocations are smaller (12-24 nodes baseline) and
the sensor-network domain doesn't need 100k nodes -- the
realistic deployment ceiling is ~10^3 nodes per fleet.
"""
import json
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp


SEED = 42


def build_sensor_network(N, seed=SEED):
    """Heterogeneous Watts-Strogatz topology.

    k = max(4, int(log2(N))) initial neighbors, p=0.15 rewire prob.
    Heterogeneous node types: 25% magnetometer, 25% UWB, 25% acoustic, 25% IR.
    """
    rng = np.random.default_rng(seed)
    k = max(4, int(np.log2(N)))
    if k % 2 == 1:
        k += 1
    G = nx.watts_strogatz_graph(N, k=k, p=0.15, seed=seed)
    # Node types
    types = rng.choice(['mag', 'uwb', 'acoustic', 'ir'], size=N)
    nx.set_node_attributes(G, dict(zip(range(N), types)), 'type')
    # Governance tiers (T1 critical, T2 important, T3 routine)
    tier_probs = [0.2, 0.4, 0.4]
    tiers = rng.choice(['T1', 'T2', 'T3'], size=N, p=tier_probs)
    nx.set_node_attributes(G, dict(zip(range(N), tiers)), 'tier')
    return G, tiers


def compute_cei_scores(G, demand, tiers, alpha=0.4, beta=0.2, gamma=0.4):
    """Compute CEI = alpha*C + beta*H + gamma*R for one slot."""
    N = G.number_of_nodes()

    # Centrality: sampled betweenness for tractability at large N
    k = max(20, min(int(5 * np.log2(N)), 100))
    bc = nx.betweenness_centrality(G, k=k, seed=42, normalized=True)
    C = np.array([bc[i] for i in range(N)])
    Cmax = C.max() if C.max() > 0 else 1.0
    C = C / Cmax

    # Entropy: from demand histogram per node (4 bins)
    H = np.zeros(N)
    for i in range(N):
        # Use demand[i] as a fake telemetry sample window
        sample = demand[i]
        hist, _ = np.histogram(sample, bins=4, range=(0.0, 1.0))
        p = hist / max(hist.sum(), 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            H[i] = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
    H /= np.log2(4)  # normalize to [0,1]

    # Governance R
    R_VAL = {'T1': 1.0, 'T2': 0.7, 'T3': 0.4}
    R = np.array([R_VAL[t] for t in tiers])

    CEI = alpha * C + beta * H + gamma * R
    return CEI, C, H, R


def simulate_allocator(G, tiers, n_slots=200, allocator='cei', seed=SEED):
    """Run an allocator over n_slots, return key metrics."""
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    G_FLOOR = {'T1': 0.70, 'T2': 0.40, 'T3': 0.10}
    floors = np.array([G_FLOOR[t] for t in tiers])

    # Pre-compute centrality once (it's structural, doesn't change per slot)
    k = max(20, min(int(5 * np.log2(N)), 100))
    bc = nx.betweenness_centrality(G, k=k, seed=42, normalized=True)
    C = np.array([bc[i] for i in range(N)])
    Cmax = C.max() if C.max() > 0 else 1.0
    C = C / Cmax

    R_VAL = {'T1': 1.0, 'T2': 0.7, 'T3': 0.4}
    R = np.array([R_VAL[t] for t in tiers])

    prev_alloc = np.full(N, 0.6)
    oscillations = 0
    gov_violations = 0
    total_waste = 0.0
    total_demand = 0.0

    # Track adaptive weights for convergence
    alpha, beta, gamma = 0.4, 0.2, 0.4
    weights_history = [(alpha, beta, gamma)]

    # Demand window for entropy
    demand_window = []
    WINDOW_SIZE = 24

    for t in range(n_slots):
        # Demand: diurnal pattern + noise + occasional spike
        base = 0.55 + 0.2 * np.sin(2 * np.pi * t / 24)
        demand = base + rng.normal(0, 0.1, N)
        if t % 50 == 0:
            spike_nodes = rng.choice(N, size=max(1, N // 20), replace=False)
            demand[spike_nodes] += 0.3
        demand = np.clip(demand, 0.05, 1.0)

        demand_window.append(demand)
        if len(demand_window) > WINDOW_SIZE:
            demand_window.pop(0)

        # Entropy from window
        if len(demand_window) >= 4:
            window_arr = np.array(demand_window)
            H = np.zeros(N)
            for i in range(N):
                hist, _ = np.histogram(window_arr[:, i], bins=4, range=(0.0, 1.0))
                p = hist / max(hist.sum(), 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    H[i] = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
            H /= np.log2(4)
        else:
            H = np.full(N, 0.5)

        if allocator == 'cei':
            CEI = alpha * C + beta * H + gamma * R
            # Allocation = scaled CEI, enforced against floors
            alloc = np.clip(CEI * 1.2, 0.0, 1.0)
            alloc = np.maximum(alloc, floors)

            # Adaptive weight update: minimize oscillation + variance
            # Gradient step on oscillation freq + weight variance
            osc_signal = np.abs(alloc - prev_alloc).mean()
            if osc_signal > 0.10:
                alpha = min(0.55, alpha + 0.01)
                beta = max(0.10, beta - 0.005)
                gamma = max(0.10, gamma - 0.005)
            else:
                alpha = max(0.30, alpha - 0.002)
                beta = min(0.35, beta + 0.001)
                gamma = min(0.45, gamma + 0.001)
            # Renormalize
            s = alpha + beta + gamma
            alpha, beta, gamma = alpha/s, beta/s, gamma/s
            weights_history.append((alpha, beta, gamma))

        elif allocator == 'reactive':
            # Threshold-based; no governance awareness
            alloc = np.where(demand > 0.7, 1.0, np.where(demand < 0.4, 0.3, 0.6))

        elif allocator == 'static':
            alloc = np.full(N, 0.85)

        # Track metrics
        changes = np.abs(alloc - prev_alloc) > 0.05
        oscillations += int(changes.sum())

        # Governance: T1 or T2 below floor
        violations = (alloc < floors) & np.array([tt in ('T1', 'T2') for tt in tiers])
        gov_violations += int(violations.sum())

        # Waste = allocated above demand
        waste = np.maximum(0, alloc - demand).sum()
        total_waste += waste
        total_demand += demand.sum()

        prev_alloc = alloc

    governance_compliance = 100.0 * (1 - gov_violations / (n_slots * N))
    waste_pct = 100.0 * total_waste / max(total_demand + total_waste, 1e-9)

    return {
        'allocator': allocator,
        'N': N,
        'n_slots': n_slots,
        'oscillations': oscillations,
        'governance_violations': gov_violations,
        'governance_compliance_pct': round(governance_compliance, 2),
        'waste_pct': round(waste_pct, 2),
        'final_weights': weights_history[-1] if allocator == 'cei' else None,
        'n_weight_updates': len(weights_history) if allocator == 'cei' else 0,
    }


def run_scaling_experiment():
    SCALES = [24, 64, 256, 1024]
    results = {}
    timing = {}

    for N in SCALES:
        print(f"\n=== Scale N={N} ===", flush=True)
        G, tiers = build_sensor_network(N, seed=SEED)
        E = G.number_of_edges()
        print(f"  Graph: |V|={N}, |E|={E}, mean_degree={2*E/N:.1f}", flush=True)

        scale_results = {}

        # Time CEI scoring (one slot)
        rng = np.random.default_rng(SEED)
        demand_window = rng.uniform(0, 1, (24, N))

        t0 = time.perf_counter()
        CEI, C, H, R = compute_cei_scores(G, demand_window.T, tiers)
        t_cei = time.perf_counter() - t0

        # Run full 200-slot simulations
        for allocator in ['cei', 'reactive', 'static']:
            t0 = time.perf_counter()
            r = simulate_allocator(G, tiers, n_slots=200, allocator=allocator, seed=SEED)
            r['wall_clock_s'] = round(time.perf_counter() - t0, 2)
            scale_results[allocator] = r
            print(f"  {allocator}: osc={r['oscillations']}, gov={r['governance_compliance_pct']}%, waste={r['waste_pct']}%, t={r['wall_clock_s']}s", flush=True)

        # Compute CEI improvement vs reactive
        cei = scale_results['cei']
        reactive = scale_results['reactive']
        if reactive['oscillations'] > 0:
            osc_reduction = 100.0 * (reactive['oscillations'] - cei['oscillations']) / reactive['oscillations']
        else:
            osc_reduction = 0.0
        if reactive['waste_pct'] > 0:
            waste_reduction = 100.0 * (reactive['waste_pct'] - cei['waste_pct']) / reactive['waste_pct']
        else:
            waste_reduction = 0.0

        scale_results['scoring_time_s'] = round(t_cei, 3)
        scale_results['edges'] = E
        scale_results['mean_degree'] = round(2*E/N, 2)
        scale_results['osc_reduction_vs_reactive_pct'] = round(osc_reduction, 2)
        scale_results['waste_reduction_vs_reactive_pct'] = round(waste_reduction, 2)

        results[str(N)] = scale_results

    with open('/home/claude/paper2_day1/paper2_mega_scale_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n\n=== Summary Table ===", flush=True)
    print(f"{'N':>6} {'|E|':>8} {'CEI osc':>8} {'React osc':>10} {'osc red%':>9} {'CEI gov%':>9} {'CEI waste%':>11} {'react waste%':>13} {'waste red%':>11} {'score time':>11}", flush=True)
    for N in SCALES:
        r = results[str(N)]
        cei = r['cei']
        react = r['reactive']
        print(f"{N:>6} {r['edges']:>8} {cei['oscillations']:>8} {react['oscillations']:>10} {r['osc_reduction_vs_reactive_pct']:>9.2f} {cei['governance_compliance_pct']:>9.2f} {cei['waste_pct']:>11.2f} {react['waste_pct']:>13.2f} {r['waste_reduction_vs_reactive_pct']:>11.2f} {r['scoring_time_s']:>10.3f}s", flush=True)

    return results


if __name__ == '__main__':
    run_scaling_experiment()
