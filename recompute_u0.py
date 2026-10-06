"""Recompute the cached coarse registration for every stored dataset."""
import os, glob, numpy as np, torch
from multiprocessing import Pool
import model

def one(y):
    torch.set_num_threads(1)
    y = torch.tensor(y)[None]; y = y / y.abs().pow(2).mean().sqrt()
    return model.init_motion(y)[0].numpy().astype(np.float16)

if __name__ == '__main__':
    files = sorted(glob.glob('data/*.npz'), key=lambda f: (('train' in f), f))
    for f in files:
        z = dict(np.load(f))
        with Pool(2) as p:
            u0 = p.map(one, list(z['y']), chunksize=2)
        z['u0'] = np.stack(u0)
        np.savez(f, **z); print('done', f, flush=True)
