import json, numpy as np, torch, data, evaluate
torch.set_num_threads(2)
D = data.load('val'); best = json.load(open('results/tuning.json'))
rows = [(p, a) for p, a in best['jpa']['table']]
for p in [dict(tau=0.003, lam=2.0), dict(tau=0.01, lam=2.0)]:
    r, _, _ = evaluate.run_baseline('jpa', p, D)
    ag = {k: float(np.nanmean([x[k] for x in r])) for k in r[0]}; rows.append((p, ag)); print(p, {k: round(v, 3) for k, v in ag.items()}, flush=True)
A = np.array([[a['auc'], a['ssim'], -a['nrmse']] for _, a in rows]); Z = (A - A.mean(0)) / (A.std(0) + 1e-9)
j = int(np.argmax(Z.mean(1))); print('BEST', rows[j][0])
best['jpa'] = dict(params=rows[j][0], val=rows[j][1], table=rows); json.dump(best, open('results/tuning.json', 'w'), indent=1)
