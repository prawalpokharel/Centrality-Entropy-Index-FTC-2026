"""
mega_scale_experiment.py — Measure CEI per-cycle wall-clock time at 10k / 50k / 100k nodes.

This is a runtime measurement of the CEI scoring pipeline on synthetic
sparse dependency graphs that match enterprise topology characteristics
(power-law degree distribution, mean degree ~ log N).

We measure:
  - Centrality recompute time (sampled betweenness + PageRank)
  - Per-step entropy estimation
  - Per-step CEI scoring
  - Governance lookup
  - Total per-cycle latency

We DO NOT run a full allocation simulation at this scale — only the
scoring pipeline, since that's the part whose computational scaling
is the open question reviewers asked about.
"""
import time
import json
import numpy as np
import networkx as nx
import scipy.sparse as sp


def build_dependency_graph(N, seed=42):
    """Build a sparse dependency graph of N nodes.

    Use a Barabasi-Albert preferential attachment graph (power-law degree),
    matching the empirical sparsity of microservice dependency graphs.
    Mean degree is ~ 2*m where m is the attachment parameter.
    """
    m = max(2, int(np.log(N)))  # scales like log N
    G = nx.barabasi_albert_graph(N, m=m, seed=seed)
    return G


def assign_tiers(N, rng):
    """Assign T1/T2/T3 tiers in 15/35/50 proportion."""
    tiers = np.full(N, 'T3', dtype='<U2')
    n_t1 = int(0.15 * N)
    n_t2 = int(0.35 * N)
    idx = rng.permutation(N)
    tiers[idx[:n_t1]] = 'T1'
    tiers[idx[n_t1:n_t1+n_t2]] = 'T2'
    return tiers


def time_centrality(G, sampling_k=None):
    """Time sampled betweenness centrality computation.

    For large N, exact betweenness is O(N*E). Sampled approximation
    with k pivot nodes runs in O(k*E) which is what we time here.
    Sampling rate is capped to keep per-cycle latency tractable.
    """
    N = G.number_of_nodes()
    if sampling_k is None:
        # Aggressive sampling for very large graphs: k = log2(N) * 5
        # This is sub-sqrt(N) but matches Riondato-style bounds for
        # epsilon-approximation at moderate confidence.
        sampling_k = max(20, min(int(5 * np.log2(N)), 200))

    t0 = time.perf_counter()
    bc = nx.betweenness_centrality(G, k=sampling_k, seed=42, normalized=True)
    t_betw = time.perf_counter() - t0

    t0 = time.perf_counter()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    t_pr = time.perf_counter() - t0

    # Combine into centrality vector
    bc_arr = np.array([bc[i] for i in range(N)])
    pr_arr = np.array([pr[i] for i in range(N)])
    C = 0.5 * bc_arr / max(bc_arr.max(), 1e-12) + 0.5 * pr_arr / max(pr_arr.max(), 1e-12)

    return C, t_betw, t_pr, sampling_k


def time_entropy_per_step(N, window_size=24, n_states=4, rng=None):
    """Time per-step Shannon entropy over a sliding telemetry window."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Pre-allocate a window of state distributions
    window = rng.integers(0, n_states, size=(window_size, N))

    t0 = time.perf_counter()
    # Compute entropy for each node
    H = np.zeros(N)
    for s in range(n_states):
        p = (window == s).sum(axis=0) / window_size
        # Shannon term, ignoring zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(p > 0, -p * np.log2(p), 0.0)
        H += term
    H /= np.log2(n_states)  # normalize to [0, 1]
    t_ent = time.perf_counter() - t0
    return H, t_ent


def time_cei_combine_and_governance(N, C, H, tiers, rng):
    """Time the CEI weighted sum, governance lookup, and floor enforcement."""
    G_FLOOR = {'T1': 0.70, 'T2': 0.40, 'T3': 0.10}
    R_VAL = {'T1': 1.0, 'T2': 0.7, 'T3': 0.4}

    t0 = time.perf_counter()
    R = np.array([R_VAL[t] for t in tiers])
    floors = np.array([G_FLOOR[t] for t in tiers])
    alpha, beta, gamma = 0.4, 0.2, 0.4
    cei = alpha * C + beta * H + gamma * R
    # Imagine alloc derived from cei, then enforced against floors
    alloc = np.clip(cei, 0.0, 1.0)
    alloc = np.maximum(alloc, floors)
    t_score = time.perf_counter() - t0
    return alloc, t_score


def run_experiment(N, seed=42):
    """Run one full timing experiment at scale N."""
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    G = build_dependency_graph(N, seed=seed)
    t_build = time.perf_counter() - t0

    E = G.number_of_edges()
    mean_deg = 2 * E / N

    # Centrality (the dominant cost)
    C, t_betw, t_pr, k_sampled = time_centrality(G)

    # Entropy (per step)
    H, t_ent = time_entropy_per_step(N, rng=rng)

    # CEI combine + governance lookup
    tiers = assign_tiers(N, rng)
    alloc, t_score = time_cei_combine_and_governance(N, C, H, tiers, rng)

    # Memory: sparse adjacency
    A = nx.to_scipy_sparse_array(G, format='csr')
    mem_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes

    return {
        'N': N,
        'E': E,
        'mean_degree': round(mean_deg, 2),
        'graph_build_s': round(t_build, 3),
        'betweenness_s': round(t_betw, 3),
        'betweenness_k_samples': k_sampled,
        'pagerank_s': round(t_pr, 3),
        'centrality_total_s': round(t_betw + t_pr, 3),
        'per_step_entropy_ms': round(t_ent * 1000, 2),
        'per_step_cei_score_ms': round(t_score * 1000, 2),
        'per_step_total_ms': round((t_ent + t_score) * 1000, 2),
        'sparse_adj_memory_mb': round(mem_bytes / (1024 * 1024), 2),
    }


if __name__ == '__main__':
    SCALES = [1000, 10000, 50000, 100000]
    results = []
    for N in SCALES:
        print(f"\n=== Running N={N:,} ===")
        r = run_experiment(N)
        for k, v in r.items():
            print(f"  {k}: {v}")
        results.append(r)

    with open('/home/claude/paper3/mega_scale_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to /home/claude/paper3/mega_scale_results.json")
