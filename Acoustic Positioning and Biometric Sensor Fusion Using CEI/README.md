# Paper 2 (FTC #140): Governance-Aware Resource Allocation for Distributed Sensor Networks

**Author:** Prawal Pokharel
**Venue:** Future Technologies Conference (FTC) 2026 — Berlin
**Publication:** Springer LNCS/LNNS proceedings (camera-ready, revision 2 post-audit)

## Audit Status

This version has been audited for:
- ✅ Mathematical correctness of all 5 propositions (Proposition 4 bound corrected post-audit)
- ✅ Numerical consistency between abstract, body, and source JSON data
- ✅ All 32 bibliography entries cited ≥2× in body
- ✅ All 10 tables introduced by prose before float
- ✅ All 4 numbered equations referenced inline
- ✅ Abstract within LNCS 250-word soft limit (243 words)
- ✅ Zero spelling errors detected
- ✅ Zero overclaim phrases ("uniquely achieves" softened in all 3 locations)
- ✅ Companion-paper citations correctly marked "FTC 2026 (companion paper, in press)"

## Files

| File | Description |
|------|-------------|
| `paper2.pdf` | Camera-ready PDF (25 pages, LNCS/LNNS) |
| `build_paper2.py` | ReportLab build script |
| `paper2_pre_revision2.pdf` | v9 baseline (pre-Prop4/5/5.5) for diff |
| `build_paper2_pre_revision2.py` | v9 build script for diff |
| `paper2_mega_scale_experiment.py` | Section 5.5 / Table 4a code |
| `paper2_mega_scale_results.json` | Real measured scaling data |
| `rl_baseline_underwater_fixed.py` | PPO baseline (Section 6.3) |
| `rl_baseline_sensor_fusion_fixed.py` | PPO baseline (Section 7.4) |
| `rl_lagrangian_ppo.py` | Lagrangian-PPO constrained-RL baseline |
| `bellhop_validation.py` | BELLHOP acoustic channel validation |

## Build

```bash
pip install reportlab
python3 build_paper2.py
```

## Reproducing Section 5.5

```bash
pip install numpy networkx scipy
python3 paper2_mega_scale_experiment.py
```

Runtime: ~30 seconds. Produces `paper2_mega_scale_results.json` with the exact numbers in Table 4a.

## Key Mathematical Content

**Proposition 4 (Regret Bound vs Oracle Allocator).** For CEI allocation $x_t$ vs an oracle $x^*_t = d_t$ under Lipschitz demand:

$$\sum_{t=1}^{T} [L(x_t, d_t) - L(x^*_t, d_t)] \le 2T(D_0^2 + L^2 H^2)$$

where $H$ is the hysteresis window, $L$ is the demand Lipschitz constant, and $D_0 = \sup_t \|x_t - d_t\|_2$ at segment-start times. Proof by triangle inequality on each H-slot segment + $(a+b)^2 \le 2a^2 + 2b^2$.

**Proposition 5 (Adaptive Weight Convergence).** Projected subgradient on simplex-constrained stability loss converges at $O(1/\sqrt{k})$:

$$L(w_k^{\text{avg}}) - L(w^*) \le \frac{G \cdot \text{diam}(\Delta)}{\sqrt{k}}$$

Standard application of Boyd & Vandenberghe Theorem 8.3.1 / Bertsekas Proposition 2.3.2.

## Verified Numerical Claims

| Claim | Source |
|-------|--------|
| 50% oscillation suppression (underwater) | Section 6, Table 4 (12-node) |
| 24.2% bandwidth savings (underwater) | Section 6, Table 4 |
| 95.6% detection availability (fusion) | Section 7, Table 6 (24-node) |
| 41% wasted bandwidth reduction | Section 7 |
| 97.3% governance compliance under jamming | Section 7 |
| 19.8 pp PPO governance drop | Section 7.4 |
| **98% scale-invariant oscillation reduction (new)** | Section 5.5, Table 4a |
| **100% governance compliance N=24 to 1024 (new)** | Section 5.5, Table 4a |

All Table 4a numbers traced exactly to `paper2_mega_scale_results.json`.

## Companion Papers

- **Paper #132** (defense infrastructure CEI)
- **This paper #140** (sensor networks CEI)
- **Paper 3** (enterprise cloud CEI)
