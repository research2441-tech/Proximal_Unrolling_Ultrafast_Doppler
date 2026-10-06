"""Training of the unrolled reconstruction network, its ablations and the learned baselines.
Usage: python train.py <variant> <seed> [iters]"""
import json
import os
import sys
import time
import numpy as np
import torch
import model
import data

torch.set_num_threads(int(os.environ.get('NTHREADS', '2')))

ABL = 'nuclear'
VARIANTS = {
    'full':        dict(cls='unrolled', flags={}, K=6),
    'no_adapt':    dict(cls='unrolled', flags=dict(adaptive=False), K=6),
    'no_struct':   dict(cls='unrolled', flags=dict(structured=False), K=6),
    'l1_sparse':   dict(cls='unrolled', flags=dict(structured=False, group=False), K=6),
    'no_psf':      dict(cls='unrolled', flags=dict(psf='none'), K=6),
    'oracle_psf':  dict(cls='unrolled', flags=dict(psf='oracle'), K=6),
    'no_motion':   dict(cls='unrolled', flags=dict(motion=False), K=6),
    'no_weights':  dict(cls='unrolled', flags=dict(weights=False), K=6),
    'nuclear':     dict(cls='unrolled', flags=dict(nuclear=True), K=6),
    'K3':          dict(cls='unrolled', flags={}, K=3),
    'K9':          dict(cls='unrolled', flags={}, K=9),
    'durpca':      dict(cls='durpca', flags={}, K=6),
    'cnn':         dict(cls='cnn', flags={}, K=0),
}


def build_model(variant):
    v = VARIANTS[variant]
    if v['cls'] == 'unrolled':
        return model.UnrolledJMTF(K=v['K'], flags=v['flags'])
    if v['cls'] == 'durpca':
        return model.DeepUnfoldedRPCA(K=v['K'])
    return model.DirectCNN()


def db(p, floor):
    return 10 * torch.log10(p + floor)


def batch(D, idx, rng, crop=32):
    H = D['y'].shape[-1]
    ys, us, u0s, pds, psfs = [], [], [], [], []
    for i in idx:
        z0 = rng.integers(2, H - crop - 2); x0 = rng.integers(2, H - crop - 2)
        sl = (slice(z0, z0 + crop), slice(x0, x0 + crop))
        ys.append(D['y'][i][:, sl[0], sl[1]]); us.append(D['u'][i][:, :, sl[0], sl[1]].astype(np.float32))
        u0s.append(D['u0'][i][:, :, sl[0], sl[1]].astype(np.float32)); pds.append(D['pd_gt'][i][sl])
        psfs.append(D['psf'][i])
    t = lambda a: torch.tensor(np.stack(a))
    return t(ys), t(us), t(u0s), t(pds)[:, None], t(psfs)


def loss_fn(out, u, pd_gt, variant):
    m = 2
    c = (slice(None), slice(None), slice(m, -m), slice(m, -m))
    fg = 1e-3 * pd_gt.flatten(1).amax(1).view(-1, 1, 1, 1)
    fh = 1e-3 * out['pd'].flatten(1).amax(1).view(-1, 1, 1, 1).detach() + 1e-20
    e = db(out['pd'], fh) - db(pd_gt, fg)
    e = e - e[c].flatten(1).median(1).values.view(-1, 1, 1, 1)      # scale-invariant dB error
    l_pd = e[c].abs().mean()
    l_mot = torch.zeros(())
    if out.get('uhist'):
        l_mot = sum((uk - u).pow(2).sum(2).add(1e-8).sqrt()[:, :, m:-m, m:-m].mean() for uk in out['uhist']) / len(out['uhist'])
    l_unc = torch.zeros(())
    if 'logvar' in out:
        ed = e.detach()[c]; lv = out['logvar'][c]
        l_unc = 0.5 * (ed ** 2 * torch.exp(-lv) + lv).mean()
    return l_pd + 10.0 * l_mot + 0.1 * l_unc, dict(pd=float(l_pd.detach()), mot=float(l_mot.detach()), unc=float(l_unc.detach()))


def main(variant, seed, iters=400, bs=None, out_dir=os.environ.get('RUN_DIR', 'runs')):
    os.makedirs(out_dir, exist_ok=True)
    if variant == 'cnn' and seed >= 3:
        print('cnn baseline uses three seeds'); return
    bs = bs or (8 if variant == 'cnn' else 4)
    tag = f'{variant}_s{seed}'
    path = os.path.join(out_dir, tag + '.pt')
    if os.path.exists(path):
        print('exists', path); return
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    D = data.load('train')
    net = build_model(variant)
    lr = 2e-3 if variant == 'cnn' else 3e-3
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    n = len(D['y']); t0 = time.time(); logs = []
    for it in range(iters):
        idx = rng.choice(n, bs, replace=False)
        y, u, u0, pd, psf = batch(D, idx, rng)
        out = net(y, psf_true=psf, u0=u0)
        loss, parts = loss_fn(out, u, pd, variant)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        logs.append(parts)
        if it % 50 == 0 or it == iters - 1:
            avg = {k: float(np.mean([l[k] for l in logs[-50:]])) for k in parts}
            print(tag, it, {k: round(v, 4) for k, v in avg.items()}, f'{time.time() - t0:.0f}s', flush=True)
    torch.save(dict(state=net.state_dict(), variant=variant, seed=seed, iters=iters,
                    train_time=time.time() - t0, logs=logs), path)


if __name__ == '__main__':
    v = sys.argv[1]; s = int(sys.argv[2])
    it = int(sys.argv[3]) if len(sys.argv) > 3 else {'durpca': 1500, 'cnn': 5000}.get(v, 400)
    main(v, s, it)
