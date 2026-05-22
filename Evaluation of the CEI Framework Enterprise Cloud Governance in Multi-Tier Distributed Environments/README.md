# Paper 3: Federated Deployment and Empirical Evaluation of the CEI Framework

**Author:** Prawal Pokharel
**Venue:** Future Technologies Conference (FTC) 2026 — Berlin
**Publication:** Springer LNCS proceedings (camera-ready, revision 4 post-audit)

## Audit Status

This version has been audited for:
- ✅ All 4 numbered equations referenced inline
- ✅ All 4 tables introduced by prose before float
- ✅ All 3 figures introduced by prose before float
- ✅ All 32 bibliography entries cited at least once
- ✅ All CI bounds match measured JSON data exactly (post-audit corrections applied)
- ✅ All scaling claims (Tables 3-4) verified arithmetically against raw measurements
- ✅ Zero spelling errors
- ✅ Governance compliance narrowly defined
- ✅ Production claims framed as single-deployment point estimates
- ✅ Operational claims softened where not directly verifiable

## Files

| File | Description |
|------|-------------|
| `paper3.pdf` | Camera-ready PDF (21 pages, LNCS, 3 figures, 4 tables) |
| `paper3.tex` | LaTeX source |
| `paper3_pre_revision.tex` | Initial submission for diff |
| `llncs.cls`, `splncs04.bst` | Springer LNCS class/style |
| `azure_calibrated_simulation.py` | Table 1 / Figure 3 counterfactual |
| `workload_prediction_lstm.py` | Table 2 LSTM forecaster |
| `compute_cis.py` | Section 7 CI computation (30 seeds) |
| `mega_scale_experiment.py` | Tables 3 and 4 scaling experiment |
| `run_all_scales.py` | Wrapper for all scales |
| `cis_results.json` | 30-seed CI measurements |
| `mega_scale_results.json` | Mega-scale timing measurements |
| `*_result.json` | Per-allocator and per-scale data |

## Build

```bash
pdflatex paper3 && pdflatex paper3
```

## Reproducing All Experimental Results

```bash
# Table 1 / Figure 3 (counterfactual): ~5 min
python3 azure_calibrated_simulation.py

# Table 2 (LSTM): ~10 sec
python3 workload_prediction_lstm.py

# Section 7 confidence intervals: ~3 min
python3 compute_cis.py

# Tables 3 and 4 (mega-scale scaling): ~3 min
python3 run_all_scales.py
```

All scripts use deterministic seeds and reproduce the exact paper numbers.

## Audit Corrections (Revision 4)

| Item | Before | After | Source |
|------|--------|-------|--------|
| 22.7% cost reduction CI lower | 22.71% | 22.69% | measured 22.6867% |
| 38.03% oscillation CI upper | 38.14% | 38.15% | measured 38.146% |
| Centrality recompute claim | "roughly every two hours" | softened to "minutes-to-hours" | unverifiable specifics removed |

## Companion Papers

- **Paper #132** (defense infrastructure CEI)
- **Paper #140** (sensor networks CEI)
- **This paper** (enterprise cloud CEI — production + scaling validation)
