"""Telemetry degradation experiments - Paper 1 Section VIII new subsection."""
import json
import time
from typing import Dict, List
from rl_baseline_nc3 import NC3Config
from cei_reference import TelemetryChannel, evaluate_cei

SEED = 2026
N_TRIALS = 20
SCENARIO = "targeted"


def main():
    cfg = NC3Config()
    conditions: List[Dict] = [
        {"label": "Nominal", "lag": 0, "noise": 0.0, "dropout": 0.0},
        {"label": "Lag 5s", "lag": 5, "noise": 0.0, "dropout": 0.0},
        {"label": "Lag 15s", "lag": 15, "noise": 0.0, "dropout": 0.0},
        {"label": "Lag 30s", "lag": 30, "noise": 0.0, "dropout": 0.0},
        {"label": "Noise 5%", "lag": 0, "noise": 0.05, "dropout": 0.0},
        {"label": "Noise 10%", "lag": 0, "noise": 0.10, "dropout": 0.0},
        {"label": "Noise 20%", "lag": 0, "noise": 0.20, "dropout": 0.0},
        {"label": "Dropout 10%", "lag": 0, "noise": 0.0, "dropout": 0.10},
        {"label": "Dropout 20%", "lag": 0, "noise": 0.0, "dropout": 0.20},
        {"label": "Dropout 30%", "lag": 0, "noise": 0.0, "dropout": 0.30},
    ]

    results = []
    t0 = time.time()
    for c in conditions:
        tc = (None if all(v == 0 for v in (c["lag"], c["noise"], c["dropout"]))
              else TelemetryChannel(lag=c["lag"], noise=c["noise"],
                                    dropout=c["dropout"], seed=SEED))
        cond_t0 = time.time()
        r = evaluate_cei(cfg, SCENARIO, telemetry=tc, n_trials=N_TRIALS)
        elapsed = time.time() - cond_t0
        results.append({**c, **r, "elapsed_s": elapsed})
        print(f"  {c['label']:>14s}: comm={r['comm_success_mean']:.3f} "
              f"gov={r['gov_compliance_mean']:.3f} ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\nTotal: {total:.1f}s")
    out = {"scenario": SCENARIO, "n_trials": N_TRIALS, "seed": SEED,
           "conditions": results, "elapsed_seconds": total}
    with open("telemetry_degradation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(f"Telemetry degradation - scenario={SCENARIO}, n_trials={N_TRIALS}")
    print("=" * 60)
    main()
