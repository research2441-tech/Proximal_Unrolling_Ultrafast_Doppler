"""Extends the validation grids where the first selection fell on a grid edge, then re-selects."""
import json, os, numpy as np, torch, data, evaluate
torch.set_num_threads(1)
best = json.load(open('results/tuning.json'))
D = data.load('val')
extra = {'jpa': [dict(tau=t, lam=l) for t in [0.0003, 0.001] for l in [2.0, 4.0]],
         'asvd': [dict(fc=f) for f in [0.16, 0.2, 0.25]]}
for name, grid in extra.items():
    rows = [(p, a) for p, a in best[name]['table']]
    for p in grid:
        r, _, _ = evaluate.run_baseline(name, p, D)
        ag = {k: float(np.nanmean([x[k] for x in r])) for k in r[0]}
        rows.append((p, ag)); print(name, p, {k: round(v, 3) for k, v in ag.items()}, flush=True)
    A = np.array([[a['auc'], a['ssim'], -a['nrmse']] for _, a in rows])
    Z = (A - A.mean(0)) / (A.std(0) + 1e-9)
    j = int(np.argmax(Z.mean(1)))
    best[name] = dict(params=rows[j][0], val=rows[j][1], table=rows)
    print('BEST', name, rows[j][0], flush=True)
json.dump(best, open('results/tuning.json', 'w'), indent=1)
