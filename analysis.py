"""Statistics, tables (as markup rows) and result figures from the evaluation JSON files."""
import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data
import metrics

RES = os.environ.get('RES_DIR', 'results')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})

BASE = ['svd', 'asvd', 'rpca', 'mcsvd', 'jpa']
LEARN = ['durpca', 'cnn', 'full']
NAMES = {'svd': 'SVD', 'asvd': 'A-SVD', 'rpca': 'RPCA', 'mcsvd': 'MC-SVD', 'jpa': 'JPA', 'durpca': 'DU-RPCA',
         'cnn': 'Direct CNN', 'full': 'Proposed', 'no_adapt': 'w/o adaptive map', 'no_struct': 'w/o oriented grouping',
         'l1_sparse': 'pixel-wise l1 shrinkage', 'no_psf': 'w/o PSF model', 'oracle_psf': 'oracle PSF',
         'no_motion': 'w/o motion model', 'no_weights': 'w/o reliability weights', 'nuclear': 'nuclear-norm SVT',
         'K3': 'K = 3', 'K9': 'K = 9'}
METRICS = ['nrmse', 'psnr', 'ssim', 'cr', 'vsnr', 'auc', 'recall', 'recall_slow', 'epe']
HIGHER = {'nrmse': False, 'epe': False}


def load(reg):
    p = os.path.join(RES, f'{reg}.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def per_seq(res, method, metric):
    """Seed-averaged per-sequence values (n_seq,) and seed-level means (n_seed,)."""
    keys = [k for k in res if k == method or k.startswith(method + '@')]
    if not keys:
        return None, None
    M = np.array([[r[metric] for r in res[k]['per_seq']] for k in keys], dtype=float)
    return np.nanmean(M, 0), np.nanmean(M, 1)


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(pvals):
    p = np.array(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def rank_biserial(a, b):
    d = a - b; d = d[(d != 0) & ~np.isnan(d)]
    if len(d) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def compare(res, ref, others, metric):
    a, _ = per_seq(res, ref, metric)
    out = {}
    ps = []
    for o in others:
        b, _ = per_seq(res, o, metric)
        ok = ~np.isnan(a) & ~np.isnan(b)
        try:
            p = wilcoxon(a[ok], b[ok]).pvalue
        except ValueError:
            p = 1.0
        ps.append(p); out[o] = dict(p=p, rb=rank_biserial(a[ok], b[ok]), delta=float(np.nanmean(a - b)))
    adj = holm(ps)
    for o, pa in zip(others, adj):
        out[o]['p_holm'] = float(pa)
    return out


def fmt(v, metric):
    if v is None or np.isnan(v):
        return '–'
    if metric in ('cr', 'vsnr', 'psnr'):
        return f'{v:.2f}'
    return f'{v:.3f}'


def main_table(reg='test'):
    res = load(reg)
    methods = [m for m in BASE + LEARN if per_seq(res, m, 'auc')[0] is not None]
    others = [m for m in methods if m != 'full']
    stats = {mt: compare(res, 'full', others, mt) for mt in METRICS if mt != 'epe'}
    rows = []
    for m in methods:
        row = {'method': m}
        for mt in METRICS:
            v, seeds = per_seq(res, m, mt)
            row[mt] = float(np.nanmean(v)); row[mt + '_ci'] = boot_ci(v)
            row[mt + '_sd'] = float(np.nanstd(seeds)) if seeds is not None and len(seeds) > 1 else None
        keys = [k for k in res if k == m or k.startswith(m + '@')]
        row['time'] = float(np.mean([res[k]['time'] for k in keys]))
        rows.append(row)
    return rows, stats


def ablation_table(reg='test'):
    res = load(reg)
    vs = [v for v in ['no_adapt', 'no_struct', 'l1_sparse', 'no_psf', 'oracle_psf', 'no_motion', 'no_weights',
                      'nuclear', 'K3', 'K9'] if per_seq(res, v, 'auc')[0] is not None]
    out = []
    st = {mt: compare(res, 'full', vs, mt) for mt in ['nrmse', 'ssim', 'auc', 'recall', 'recall_slow']}
    for v in ['full'] + vs:
        row = {'variant': v}
        for mt in ['nrmse', 'ssim', 'auc', 'recall', 'recall_slow', 'epe']:
            a, seeds = per_seq(res, v, mt)
            row[mt] = float(np.nanmean(a)); row[mt + '_sd'] = float(np.nanstd(seeds)) if len(seeds) > 1 else None
        out.append(row)
    return out, st


def sweep(regs, methods, metric):
    out = {}
    for m in methods:
        vals, cis = [], []
        for r in regs:
            v, _ = per_seq(load(r), m, metric)
            vals.append(np.nan if v is None else np.nanmean(v)); cis.append(boot_ci(v) if v is not None else (np.nan, np.nan))
        out[m] = (np.array(vals), np.array(cis))
    return out


STY = {'svd': ('o', ':', '0.55'), 'asvd': ('v', ':', '0.45'), 'rpca': ('^', '-.', '0.6'), 'mcsvd': ('s', '--', '0.35'),
       'jpa': ('D', '--', '0.2'), 'durpca': ('<', '-.', '0.4'), 'cnn': ('>', ':', '0.3'), 'full': ('*', '-', '0.0')}


def fig_sweeps(path):
    methods = BASE + ['durpca', 'cnn', 'full']
    panels = [(['motion0', 'motion05', 'motion10', 'motion15', 'motion20'], [0, 0.5, 1.0, 1.5, 2.0], 'Motion amplitude (rigid component, px)'),
              (['bnr5', 'bnr10', 'bnr15', 'bnr20'], [5, 10, 15, 20], 'Blood-to-noise ratio (dB)'),
              (['ctr20', 'ctr30', 'ctr40', 'ctr45'], [20, 30, 40, 45], 'Clutter-to-blood ratio (dB)')]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3))
    for j, (regs, xs, xl) in enumerate(panels):
        for i, mt in enumerate(['auc', 'nrmse']):
            ax = axes[i, j]
            sw = sweep(regs, methods, mt)
            for m in methods:
                v, ci = sw[m]
                mk, ls, c = STY[m]
                ax.plot(xs, v, ls=ls, marker=mk, color=c, lw=1.4 if m == 'full' else 0.9, ms=5 if m == 'full' else 3.5,
                        label=NAMES[m], zorder=5 if m == 'full' else 2)
                if m in ('full', 'jpa', 'mcsvd'):
                    ax.fill_between(xs, ci[:, 0], ci[:, 1], color=c, alpha=0.12, lw=0)
            ax.set_xlabel(xl if i == 1 else '')
            ax.set_ylabel('AUC' if mt == 'auc' else 'NRMSE')
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_xticks(xs)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=8, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2, w_pad=1.0)
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_qualitative(path, idx=(3, 17, 40)):
    maps = np.load(os.path.join(RES, 'maps_test.npz'))
    D = data.load('test')
    cols = [('GT', None), ('svd', 'svd'), ('mcsvd', 'mcsvd'), ('jpa', 'jpa'), ('durpca', 'durpca@0'), ('cnn', 'cnn@0'),
            ('full', 'full@0')]
    ex = np.load(os.path.join(RES, 'extra_test_full@0.npz'))
    fig, axes = plt.subplots(len(idx), len(cols) + 1, figsize=(7.1, 0.98 * len(idx) + 0.35))
    for r, i in enumerate(idx):
        for c, (nm, key) in enumerate(cols):
            ax = axes[r, c]
            img = (D['pd_gt'][i] if key is None else maps[key][i])[4:-4, 4:-4]
            ref = img.max() if key is None else np.percentile(img, 99.5)
            ax.imshow(10 * np.log10(np.maximum(img, 0) / ref + 1e-12), cmap='gray', vmin=-40, vmax=0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title('Reference' if key is None else NAMES[nm], fontsize=7)
        ax = axes[r, -1]
        sd = np.exp(0.5 * ex['logvar'][i])[4:-4, 4:-4]
        im = ax.imshow(sd, cmap='gray_r', vmin=0, vmax=np.percentile(sd, 99))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(r'Proposed $\hat{\upsilon}$ (dB)', fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.25)
    fig.savefig(path, dpi=400); plt.close(fig)


if __name__ == '__main__':
    rows, st = main_table()
    for r in rows:
        print(r['method'], {k: round(r[k], 3) for k in METRICS}, round(r['time'], 3))
    print(json.dumps({mt: {o: {k: round(v, 4) for k, v in d.items()} for o, d in s.items()} for mt, s in st.items()}, indent=0)[:3000])
