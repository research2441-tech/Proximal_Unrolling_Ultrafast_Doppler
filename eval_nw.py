import json, os, glob, sys, numpy as np, torch, data, evaluate
torch.set_num_threads(1)
for reg in sys.argv[1:]:
    path = f'results/{reg}.json'
    D = data.load(reg)
    out = {}
    for s in [0, 1, 2]:
        r, mp, ex, t = evaluate.run_learned('no_weights', s, D)
        out[f'no_weights@{s}'] = dict(per_seq=r, time=t)
        print(reg, s, {k: round(float(np.nanmean([x[k] for x in r])), 3) for k in ['nrmse', 'ssim', 'auc', 'recall']}, flush=True)
    json.dump(out, open(f'results/nw_{reg}.json', 'w'))
