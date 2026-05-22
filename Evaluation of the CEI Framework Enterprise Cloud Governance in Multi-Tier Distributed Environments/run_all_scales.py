import sys
sys.path.insert(0, '/home/claude/paper3')
from mega_scale_experiment import run_experiment
import json

SCALES = [1000, 10000, 50000, 100000]
results = []
for N in SCALES:
    print(f'\n=== N={N:,} ===', flush=True)
    r = run_experiment(N)
    for k, v in r.items():
        print(f'  {k}: {v}', flush=True)
    results.append(r)

with open('/home/claude/paper3/mega_scale_results.json','w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved', flush=True)
