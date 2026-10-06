"""Evaluation of all methods on validation / test / stress regimes.
Usage: python evaluate.py tune            (baseline hyper-parameter selection on validation)
       python evaluate.py run <regime>... (all methods on the listed regimes)"""
import json
import os
import sys
import time
import glob
import numpy as np
import torch
import data
import metrics
import baselines
import train

torch.set_num_threads(int(os.environ.get('NTHREADS', '2')))
RUN_DIR = os.environ.get('RUN_DIR', 'runs')
RES_DIR = os.environ.get('RES_DIR', 'results')
os.makedirs(RES_DIR, exist_ok=True)


def tensors(D, i0, i1):
    y = torch.tensor(D['y'][i0:i1]); u0 = torch.tensor(D['u0'][i0:i1].astype(np.float32))
    psf = torch.tensor(D['psf'][i0:i1])
    return y, u0, psf


def score(pd, D, i, u_hat=None):
    d = dict(pd_gt=D['pd_gt'][i], vessel=D['vessel'][i], micro=D['micro'][i], slow=D['slow'][i])
    m = metrics.image_metrics(pd, d)
    m['epe'] = metrics.motion_epe(u_hat, D['u'][i].astype(np.float32)) if u_hat is not None else float('nan')
    return m


def run_baseline(name, params, D, bs=4):
    out, maps, t_all = [], [], 0.0
    n = len(D['y'])
    for i0 in range(0, n, bs):
        y, u0, psf = tensors(D, i0, min(n, i0 + bs))
        t = time.time(); uh = None
        with torch.no_grad():
            if name == 'svd':
                pd = baselines.svd_filter(y, params['k'])
            elif name == 'asvd':
                pd = baselines.asvd_filter(y, params['fc'])
            elif name == 'rpca':
                pd = baselines.rpca(y, params['c'])
            elif name == 'mcsvd':
                pd = baselines.mc_svd(y, u0, params['k']); uh = u0
            elif name == 'jpa':
                pd, uh = baselines.jpa(y, u0, params['tau'], params['lam'])
        t_all += time.time() - t
        for j in range(pd.shape[0]):
            out.append(score(pd[j, 0].numpy(), D, i0 + j, None if uh is None else uh[j].numpy()))
            maps.append(pd[j, 0].numpy())
    return out, maps, t_all / n


def run_learned(variant, seed, D, bs=4, keep=False):
    ck = torch.load(os.path.join(RUN_DIR, f'{variant}_s{seed}.pt'), weights_only=False)
    net = train.build_model(variant); net.load_state_dict(ck['state']); net.eval()
    out, maps, extra, t_all = [], [], [], 0.0
    n = len(D['y'])
    for i0 in range(0, n, bs):
        y, u0, psf = tensors(D, i0, min(n, i0 + bs))
        t = time.time()
        with torch.no_grad():
            o = net(y, psf_true=psf, u0=u0)
        t_all += time.time() - t
        for j in range(y.shape[0]):
            uh = o['u'][j].numpy() if o.get('u') is not None else None
            out.append(score(o['pd'][j, 0].numpy(), D, i0 + j, uh))
            maps.append(o['pd'][j, 0].numpy())
            if keep:
                e = dict(sigma=None if o.get('sigma') is None else o['sigma'][j].numpy().tolist())
                if 'logvar' in o:
                    e['logvar'] = o['logvar'][j, 0].numpy()
                if 'lam' in o and o['lam'] is not None:
                    e['lam'] = o['lam'][j, 0].numpy()
                extra.append(e)
    return out, maps, extra, t_all / n


GRIDS = {
    'svd': [dict(k=k) for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]],
    'asvd': [dict(fc=f) for f in [0.02, 0.04, 0.06, 0.08, 0.1, 0.13]],
    'rpca': [dict(c=c) for c in [0.5, 1.0, 2.0, 4.0, 8.0]],
    'mcsvd': [dict(k=k) for k in [1, 2, 3, 4, 5, 6, 8]],
    'jpa': [dict(tau=t, lam=l) for t in [0.001, 0.003, 0.01] for l in [0.03, 0.1, 0.3, 1.0]],
}
SEL = 'auc'   # selection criterion on validation: mean of AUC and SSIM ranks


def tune():
    D = data.load('val')
    best = {}
    for name, grid in GRIDS.items():
        rows = []
        for p in grid:
            r, _, _ = run_baseline(name, p, D)
            ag = {k: float(np.nanmean([x[k] for x in r])) for k in r[0]}
            rows.append((p, ag)); print(name, p, {k: round(v, 3) for k, v in ag.items()}, flush=True)
        # composite: average of normalised AUC, SSIM and (1 - NRMSE)
        A = np.array([[a['auc'], a['ssim'], -a['nrmse']] for _, a in rows])
        Z = (A - A.mean(0)) / (A.std(0) + 1e-9)
        j = int(np.argmax(Z.mean(1)))
        best[name] = dict(params=rows[j][0], val=rows[j][1], table=[(p, a) for p, a in rows])
        print('BEST', name, rows[j][0], flush=True)
    json.dump(best, open(os.path.join(RES_DIR, 'tuning.json'), 'w'), indent=1)


LEARNED = ['full', 'no_adapt', 'no_struct', 'l1_sparse', 'no_psf', 'oracle_psf', 'no_motion', 'no_weights',
           'nuclear', 'K3', 'K9', 'durpca', 'cnn']


def run(regimes, save_maps=('test', 'val')):
    best = json.load(open(os.path.join(RES_DIR, 'tuning.json')))
    for reg in regimes:
        path = os.path.join(RES_DIR, f'{reg}.json')
        res = json.load(open(path)) if os.path.exists(path) else {}
        D = data.load(reg)
        maps = {}
        for name in ['svd', 'asvd', 'rpca', 'mcsvd', 'jpa']:
            if name in res:
                continue
            r, mp, t = run_baseline(name, best[name]['params'], D)
            res[name] = dict(per_seq=r, time=t); maps[name] = mp
            print(reg, name, {k: round(float(np.nanmean([x[k] for x in r])), 3) for k in r[0]}, f'{t:.2f}s', flush=True)
        allowed = LEARNED if reg in ('test', 'val') else ['full', 'durpca', 'cnn'] + (['no_psf', 'oracle_psf'] if reg == 'psfwide' else [])
        for v in allowed:
            for ck in sorted(glob.glob(os.path.join(RUN_DIR, f'{v}_s*.pt'))):
                seed = int(ck.split('_s')[-1].split('.')[0])
                key = f'{v}@{seed}'
                if key in res:
                    continue
                keep = (v == 'full')
                r, mp, ex, t = run_learned(v, seed, D, keep=keep)
                res[key] = dict(per_seq=r, time=t); maps[key] = mp
                if keep:
                    np.savez_compressed(os.path.join(RES_DIR, f'extra_{reg}_{key}.npz'),
                                        logvar=np.stack([e['logvar'] for e in ex]) if 'logvar' in ex[0] else 0,
                                        lam=np.stack([e['lam'] for e in ex]) if 'lam' in ex[0] else 0,
                                        sigma=np.array([e['sigma'] for e in ex], dtype=float))
                print(reg, key, {k: round(float(np.nanmean([x[k] for x in r])), 3) for k in r[0]}, f'{t:.2f}s', flush=True)
        json.dump(res, open(path, 'w'))
        if reg in save_maps and maps:
            old = os.path.join(RES_DIR, f'maps_{reg}.npz')
            prev = dict(np.load(old)) if os.path.exists(old) else {}
            prev.update({k: np.stack(v).astype(np.float32) for k, v in maps.items()})
            np.savez_compressed(old, **prev)


if __name__ == '__main__':
    if sys.argv[1] == 'tune':
        tune()
    else:
        run(sys.argv[2:])
