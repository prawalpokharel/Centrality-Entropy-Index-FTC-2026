"""
compute_cis.py — Per-allocator multi-seed evaluation for 95% confidence intervals.

Runs 30 independent seed-offsets per allocator (CEI, Static, Reactive).
PPO and Lagrangian-PPO are skipped here because training each seed
~30 times would take hours; their CIs are reported separately from
the existing single-seed training run.

Output: cis_results.json with mean and 95% CI for cost, oscillation
reduction, utilization, governance compliance for each allocator.
"""
import sys
import os
import json
import importlib.util
import numpy as np

# Load the simulation module's globals (skip main())
spec = importlib.util.spec_from_file_location("acs", "/home/claude/paper3/azure_calibrated_simulation.py")
acs = importlib.util.module_from_spec(spec)
src = open('/home/claude/paper3/azure_calibrated_simulation.py').read()
src_pre_main = src.split('def main(')[0]
exec(src_pre_main, acs.__dict__)

ALLOCATORS = [
    ('static', acs.alloc_static),
    ('reactive', acs.alloc_reactive),
    ('cei', None),  # CEI branches internally
]

N_SEEDS = 30
print(f"Running {N_SEEDS} independent seeds per allocator...")

all_results = {}
for name, fn in ALLOCATORS:
    print(f"\n--- {name} ---")
    costs = []
    osc = []
    util = []
    gov = []
    for s in range(N_SEEDS):
        r = acs.evaluate_allocator(name, fn, n_episodes=1, seed=42 + s)
        costs.append(r.monthly_cost_usd)
        osc.append(r.oscillation_events)
        util.append(r.utilization_efficiency_pct)
        gov.append(r.governance_compliance_pct)
    costs = np.array(costs)
    osc = np.array(osc)
    util = np.array(util)
    gov = np.array(gov)

    def ci(arr):
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        sem = s / np.sqrt(len(arr))
        # 95% CI, n=30 -> t ≈ 2.045
        t = 2.045
        return {'mean': m, 'std': s, 'ci95_lo': m - t*sem, 'ci95_hi': m + t*sem,
                'half_width': t*sem}

    all_results[name] = {
        'monthly_cost_usd': ci(costs),
        'oscillation_events': ci(osc),
        'utilization_efficiency_pct': ci(util),
        'governance_compliance_pct': ci(gov),
        'n_seeds': N_SEEDS,
    }
    print(f"  cost ${all_results[name]['monthly_cost_usd']['mean']:.0f} "
          f"± ${all_results[name]['monthly_cost_usd']['half_width']:.0f}")
    print(f"  osc  {all_results[name]['oscillation_events']['mean']:.1f} "
          f"± {all_results[name]['oscillation_events']['half_width']:.1f}")

# Now compute DERIVED CIs: cost reduction % vs reactive, oscillation reduction % vs reactive
reactive_cost_mean = all_results['reactive']['monthly_cost_usd']['mean']
reactive_osc_mean = all_results['reactive']['oscillation_events']['mean']

# Per-seed pairwise differences for valid CI on reduction percentages
costs_react = []
osc_react = []
costs_cei = []
osc_cei = []
for s in range(N_SEEDS):
    r_react = acs.evaluate_allocator('reactive', acs.alloc_reactive, n_episodes=1, seed=42+s)
    r_cei = acs.evaluate_allocator('cei', None, n_episodes=1, seed=42+s)
    costs_react.append(r_react.monthly_cost_usd)
    osc_react.append(r_react.oscillation_events)
    costs_cei.append(r_cei.monthly_cost_usd)
    osc_cei.append(r_cei.oscillation_events)

cost_red_pct = 100.0 * (np.array(costs_react) - np.array(costs_cei)) / np.array(costs_react)
osc_red_pct = 100.0 * (np.array(osc_react) - np.array(osc_cei)) / np.array(osc_react)

def ci_arr(arr, name):
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    sem = s / np.sqrt(len(arr))
    t = 2.045
    return {'mean': m, 'std': s, 'half_width': t*sem,
            'ci95_lo': m - t*sem, 'ci95_hi': m + t*sem}

all_results['cei_vs_reactive_cost_reduction_pct'] = ci_arr(cost_red_pct, 'cost_red_pct')
all_results['cei_vs_reactive_oscillation_reduction_pct'] = ci_arr(osc_red_pct, 'osc_red_pct')

print(f"\n=== Derived CIs ===")
print(f"  CEI vs Reactive cost reduction: "
      f"{all_results['cei_vs_reactive_cost_reduction_pct']['mean']:.2f}% "
      f"± {all_results['cei_vs_reactive_cost_reduction_pct']['half_width']:.2f}%")
print(f"  CEI vs Reactive osc reduction:  "
      f"{all_results['cei_vs_reactive_oscillation_reduction_pct']['mean']:.2f}% "
      f"± {all_results['cei_vs_reactive_oscillation_reduction_pct']['half_width']:.2f}%")

with open('/home/claude/paper3/cis_results.json','w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved /home/claude/paper3/cis_results.json")
