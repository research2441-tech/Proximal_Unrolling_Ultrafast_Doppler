"""Dataset generation (training / validation / test regimes) with cached coarse registration."""
import os
import sys
import numpy as np
import torch
from multiprocessing import Pool
import sim
import model

ROOT = os.environ.get('DATA_DIR', 'data')

# test regimes: name -> (seed offset, n, simulator overrides)
REGIMES = {
    'val':        (900000, 32, {}),
    'test':       (100000, 64, {}),
    'motion0':    (200000, 24, dict(rigid_amp=0.0, nonrigid_amp=0.0, strain=0.0)),
    'motion05':   (210000, 24, dict(rigid_amp=0.5, nonrigid_amp=0.25, strain=0.005)),
    'motion10':   (220000, 24, dict(rigid_amp=1.0, nonrigid_amp=0.5, strain=0.01)),
    'motion15':   (230000, 24, dict(rigid_amp=1.5, nonrigid_amp=0.75, strain=0.012)),
    'motion20':   (240000, 24, dict(rigid_amp=2.0, nonrigid_amp=1.0, strain=0.015)),
    'bnr5':       (300000, 24, dict(bnr_db=5.0)),
    'bnr10':      (310000, 24, dict(bnr_db=10.0)),
    'bnr15':      (320000, 24, dict(bnr_db=15.0)),
    'bnr20':      (330000, 24, dict(bnr_db=20.0)),
    'ctr20':      (340000, 24, dict(ctr_db=20.0)),
    'ctr30':      (350000, 24, dict(ctr_db=30.0)),
    'ctr40':      (360000, 24, dict(ctr_db=40.0)),
    'ctr45':      (370000, 24, dict(ctr_db=45.0)),
    'psfwide':    (400000, 24, dict(sig_z=(2.0, 2.4), sig_x=(2.8, 3.4))),
    'tubes':      (500000, 24, dict(geometry='tubes', n_vessels=(4, 7), rigid_amp=(0.0, 0.6), nonrigid_amp=(0.0, 0.2))),
    'penetrating': (510000, 24, dict(geometry='penetrating', n_vessels=(8, 14), hf_motion=0.6)),
    'decor':      (600000, 24, dict(decor=1e-4)),
    'slowflow':   (610000, 24, dict(speed=(0.005, 0.05))),
}


def _gen(args):
    seed, over = args
    torch.set_num_threads(1)
    d = sim.simulate(seed, **over)
    y = torch.tensor(d['y'])[None]
    y = y / y.abs().pow(2).mean().sqrt()
    u0 = model.init_motion(y)[0].numpy()
    return dict(y=d['y'], u=d['u'].astype(np.float16), u0=u0.astype(np.float16), pd_gt=d['pd_gt'],
                vessel=d['vessel'], micro=d['micro'], slow=d['slow'], psf=np.array(d['psf'], np.float32),
                ctr=d['ctr'], bnr=d['bnr'], noise_std=d['noise_std'])


def build(name, seeds, over, workers=2):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, name + '.npz')
    if os.path.exists(path):
        return path
    with Pool(workers) as p:
        items = p.map(_gen, [(s, over) for s in seeds], chunksize=4)
    out = {k: np.stack([it[k] for it in items]) for k in items[0]}
    np.savez(path, **out)
    return path


def load(name):
    z = np.load(os.path.join(ROOT, name + '.npz'))
    return {k: z[k] for k in z.files}


def as_list(D):
    n = len(D['y'])
    return [dict(y=D['y'][i], u=D['u'][i].astype(np.float32), pd_gt=D['pd_gt'][i], vessel=D['vessel'][i],
                 micro=D['micro'][i], slow=D['slow'][i], psf=tuple(D['psf'][i])) for i in range(n)]


if __name__ == '__main__':
    which = sys.argv[1:] or ['train', 'val', 'test']
    for w in which:
        if w == 'train':
            build('train', list(range(1, 481)), {})
        elif w == 'stress':
            for k, (off, n, over) in REGIMES.items():
                if k not in ('val', 'test'):
                    build(k, list(range(off, off + n)), over)
                    print('built', k, flush=True)
        else:
            off, n, over = REGIMES[w]
            build(w, list(range(off, off + n)), over)
        print('built', w, flush=True)
