# Centrality-Entropy Index (CEI) — FTC 2026

Reference implementations and reproducibility code for three Future Technologies Conference (FTC) 2026 papers introducing the **Centrality-Entropy Index (CEI)**, a unified governance-aware dynamic resource allocation framework evaluated across operationally distinct domains.

The CEI framework integrates three orthogonal signals into a single composite score for each node in a dependency graph:

- **Centrality** (`C_i`): structural importance derived from betweenness centrality or PageRank
- **Entropy** (`H_i`): Shannon entropy of the demand distribution over a sliding window
- **Governance** (`R_i`): policy-driven priority based on mission-criticality, compliance, and tier classification

The composite index is computed as:

```
CEI_i(t) = α(t) · C_i(t) + β(t) · H_i(t) + γ(t) · R_i(t)
```

with adaptive weights `(α, β, γ)` updated by projected gradient descent on a stability loss. Governance floors are enforced as hard constraints in the allocation step.

---

## Repository Structure

```
.
├── defense-infrastructure/        # Strategic Communications, Cloud Resilience and Interceptor Engagement Using CEI (FTC 2026)
├── sensor-networks/               # Acoustic Positioning and Biometric Sensor Fusion Using CEI (FTC 2026)
├── enterprise-cloud-governance/   # Evaluation of the CEI Framework Enterprise Cloud Governance in Multi-Tier Distributed Environments (FTC 2026)
└── README.md                      # this file
```

Each folder contains the camera-ready PDF, LaTeX source, reproducibility code, and a paper-specific README.

| Folder | Paper | Domains Evaluated |
|--------|-------|-------------------|
| `defense-infrastructure/` | Strategic Communications, Cloud Resilience and Interceptor Engagement Using CEI | NC3 strategic communications, cloud failover under kinetic attack, multi-salvo missile defense |
| `sensor-networks/` | Acoustic Positioning and Biometric Sensor Fusion Using CEI | Underwater acoustic positioning (12-node Thorp model), multi-modal biometric sensor fusion (24-node) |
| `enterprise-cloud-governance/` | Evaluation of the CEI Framework Enterprise Cloud Governance in Multi-Tier Distributed Environments | Production Azure deployment (90-day), calibrated counterfactual vs PPO/Lagrangian-PPO, mega-scale validation at N=100k |

---

## How These Experiments Were Conducted

All experimental results in the three papers are reproducible from the code in this repository. No proprietary data is required. Simulations are deterministic under fixed random seeds.

Hardware used for the reported wall-clock numbers: single CPU core (Apple M2 Pro, 16 GB RAM). All experiments complete in under 30 minutes total on commodity hardware.

Software stack: Python 3.12, NumPy 2.x, SciPy 1.11+, NetworkX 3.6+, PyTorch 2.x (for LSTM and PPO baselines), Stable-Baselines3 2.x, ReportLab 4.x (for ReportLab paper builds), TeX Live 2023+ (for LaTeX papers).

---

## Reproducing the Results — Step by Step

### Prerequisites

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Set up a Python environment
python3 -m venv venv
source venv/bin/activate              # macOS / Linux
# venv\Scripts\activate               # Windows

# Install dependencies
pip install numpy scipy networkx torch stable-baselines3 gymnasium reportlab matplotlib
```

For LaTeX papers, install a TeX distribution:

- macOS: `brew install --cask mactex` (or BasicTeX for a smaller install)
- Linux: `sudo apt install texlive-full`
- Windows: install MiKTeX or TeX Live from https://tug.org/texlive/

### Strategic Communications, Cloud Resilience and Interceptor Engagement Using CEI

```bash
cd defense-infrastructure

# Compile the camera-ready PDF
pdflatex paper.tex
pdflatex paper.tex   # second pass for cross-references

# Run the AWS Health Dashboard calibration simulation (Section 5)
cd aws_calibration
python3 aws_incident_calibration.py
# Output: aws_calibration_results.json with throughput, governance,
# and oscillation numbers matching Table 4 in the paper.

# Run the NC3 PPO baseline (Appendix D)
cd ..
python3 rl_baseline_nc3.py
python3 run_ppo.py

# Run the degraded telemetry experiments (Appendix B)
python3 telemetry_degradation_nc3.py
```

### Acoustic Positioning and Biometric Sensor Fusion Using CEI

```bash
cd sensor-networks

# Regenerate the camera-ready PDF (ReportLab build)
python3 build_paper.py
# Output: paper.pdf (25 pages, Springer LNCS/LNNS format)

# Run the mega-scale scaling experiment (Section 5.5, Table 4a)
python3 paper2_mega_scale_experiment.py
# Output: paper2_mega_scale_results.json with measured oscillation counts,
# governance compliance, and wall-clock timing at N = 24, 64, 256, 1024.

# Run RL baselines (Sections 6.3 and 7.4)
python3 rl_baseline_underwater_fixed.py
python3 rl_baseline_sensor_fusion_fixed.py
python3 rl_lagrangian_ppo.py

# Optional: BELLHOP acoustic channel validation
python3 bellhop_validation.py
```

### Evaluation of the CEI Framework Enterprise Cloud Governance in Multi-Tier Distributed Environments

```bash
cd enterprise-cloud-governance

# Compile the camera-ready PDF
pdflatex paper.tex
pdflatex paper.tex

# Run the calibrated counterfactual (Table 1, Figure 3)
python3 azure_calibrated_simulation.py
# Output: azure_calibrated_results.json with five-allocator comparison.

# Run the LSTM workload forecaster (Table 2)
python3 workload_prediction_lstm.py
# Output: workload_prediction_results.json with skill scores across horizons.

# Compute 30-seed confidence intervals (Section 7)
python3 compute_cis.py
# Output: cis_results.json with measured CIs (22.7% ± 0.03 pp cost reduction,
# 38.0% ± 0.11 pp oscillation reduction).

# Run the mega-scale scaling experiment (Tables 3 and 4, N up to 100k)
python3 run_all_scales.py
# Output: mega_scale_results.json with centrality recompute, per-step cost,
# and sparse adjacency memory at N = 60, 1k, 10k, 50k, 100k.
```

### Expected Wall-Clock Runtimes

| Experiment | Runtime (single CPU core) |
|------------|---------------------------|
| Strategic Communications, Cloud Resilience and Interceptor Engagement — AWS calibration | ~10 min |
| Strategic Communications, Cloud Resilience and Interceptor Engagement — PPO baseline training | ~15 min |
| Strategic Communications, Cloud Resilience and Interceptor Engagement — Telemetry degradation | ~3 min |
| Acoustic Positioning and Biometric Sensor Fusion — Mega-scale scaling (N up to 1024) | ~30 sec |
| Acoustic Positioning and Biometric Sensor Fusion — RL baselines training | ~20 min |
| Evaluation of the CEI Framework Enterprise Cloud Governance — Calibrated counterfactual | ~5 min |
| Evaluation of the CEI Framework Enterprise Cloud Governance — LSTM forecaster | ~10 sec |
| Evaluation of the CEI Framework Enterprise Cloud Governance — 30-seed CIs | ~3 min |
| Evaluation of the CEI Framework Enterprise Cloud Governance — Mega-scale up to N=100k | ~3 min |

All scripts use deterministic seeds. Numbers reported in the papers reproduce exactly (within floating-point rounding) from the included code.

---

## Citation

If you use this work, please cite the relevant FTC 2026 paper:

```bibtex
@inproceedings{pokharel2026defense,
 author    = {Pokharel, Prawal},
 title     = {Governance-Aware Resource Allocation for Defense Infrastructure Under Adversarial Threat
              Strategic Communications, Cloud Resilience and Interceptor
              Engagement Using the Centrality-Entropy Index},
 booktitle = {Proceedings of the Future Technologies Conference (FTC) 2026},
 publisher = {Springer},
 series    = {Lecture Notes in Networks and Systems},
 year      = {2026}
}

@inproceedings{pokharel2026sensors,
 author    = {Pokharel, Prawal},
 title     = {Governance-Aware Resource Allocation for Distributed Sensor Networks in GPS-Denied and Contested Environments:
              Underwater Acoustic Positioning and Biometric Sensor Fusion Using the Centrality-Entropy Index},
 booktitle = {Proceedings of the Future Technologies Conference (FTC) 2026},
 publisher = {Springer},
 series    = {Lecture Notes in Networks and Systems},
 year      = {2026}
}

@inproceedings{pokharel2026cloud,
 author    = {Pokharel, Prawal},
 title     = {Federated Deployment and Empirical Evaluation of the Centrality-Entropy Index Framework for
              Enterprise Cloud Governance in Multi-Tier Distributed Environments},
 booktitle = {Proceedings of the Future Technologies Conference (FTC) 2026},
 publisher = {Springer},
 series    = {Lecture Notes in Networks and Systems},
 year      = {2026}
}
```

---

## Related Work

### Patent

The CEI framework is protected under **USPTO non-provisional patent application 19/641,446** (filed April 7, 2026), covering the governance-aware dynamic resource allocation methodology and associated architectural modules. A continuation-in-part for the `CEI_fragility` variant is in preparation.

### Production Platform

A production deployment of the CEI framework is operational at **[cloudoptimizer.app](https://cloudoptimizer.app)** — a full-stack SaaS platform implementing the twelve architectural modules described in the patent specification. The platform serves as the operational reference implementation against which the academic simulations in this repository are calibrated.

### Author

**Prawal Pokharel** — Independent Researcher, Dallas–Fort Worth, TX, USA
Email: prawal@cloudoptimizer.app
Platform: CloudOptimizer(https://cloudoptimizer.app)

---

## License

Source code in this repository is released under the MIT License. The paper PDFs and LaTeX sources are provided for reproducibility and academic citation; the published versions in Springer proceedings are subject to Springer's copyright terms.

---

## Acknowledgments

This work builds on decades of foundational research in graph centrality (Brandes; Freeman; Page, Brin et al.), information theory (Shannon), constrained optimization (Boyd & Vandenberghe; Bertsekas), and large-scale cluster management (Burns, Verma, et al. — Borg / Omega / Kubernetes). The defense-domain analysis aligns with three active U.S. Department of Defense mandates: the Joint All-Domain Command and Control (JADC2) Implementation Plan, the zero trust architecture initiative, and the 2023 DoD Data, Analytics, and AI Adoption Strategy.

The cloud-domain analysis is informed by the March 2026 Iranian drone strikes on Amazon Web Services data centers in the UAE and Bahrain, which exposed governance gaps in standard cloud disaster recovery and motivated the AWS Health Dashboard calibration in the defense infrastructure paper.
