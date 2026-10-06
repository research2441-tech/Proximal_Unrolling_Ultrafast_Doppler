"""Uncertainty calibration, stage-wise behaviour, learned parameters, PSF accuracy and the
corresponding figures."""
import glob
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
import data
import metrics
import train
import evaluate

RES = 'results'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.linewidth': 0.7})
SEEDS = sorted(int(p.split('_s')[-1][:-3]) for p in glob.glob('runs/full_s*.pt'))


def errors(reg, seed):
    maps = np.load(os.path.join(RES, f'maps_{reg}.npz'))[f'full@{seed}']
    ex = np.load(os.path.join(RES, f'extra_{reg}_full@{seed}.npz'))
    D = data.load(reg)
    E, S, ENS = [], [], []
    c = (slice(metrics.M, -metrics.M), slice(metrics.M, -metrics.M))
    for i in range(len(maps)):
        d = dict(pd_gt=D['pd_gt'][i], vessel=D['vessel'][i], micro=D['micro'][i], slow=D['slow'][i])
        E.append(metrics.db_error(maps[i], d))
        S.append(np.exp(0.5 * ex['logvar'][i])[c])
    return np.array(E), np.array(S)


def ensemble_std(reg):
    maps = np.load(os.path.join(RES, f'maps_{reg}.npz'))
    c = (slice(metrics.M, -metrics.M), slice(metrics.M, -metrics.M))
    A = np.stack([[metrics.to_db01(m)[c] * metrics.DR for m in maps[f'full@{s}']] for s in SEEDS])
    return A.std(0)


def uncertainty():
    out = {}
    Ev, Sv = errors('val', SEEDS[0]); Et, St = errors('test', SEEDS[0])
    Ens_v = ensemble_std('val'); Ens_t = ensemble_std('test')
    for name, sv, st in [('head', Sv, St), ('ensemble', Ens_v + 1e-3, Ens_t + 1e-3)]:
        cs = np.sqrt(np.mean(Ev ** 2 / sv ** 2))                       # variance scaling on validation
        z_v = (np.abs(Ev) / (cs * sv)).ravel()
        n = z_v.size; q = np.quantile(z_v, min(1, np.ceil((n + 1) * 0.9) / n))
        # acquisition-level conformal quantile
        per = np.array([np.quantile(np.abs(Ev[i]) / (cs * sv[i]), 0.9) for i in range(len(Ev))])
        m = len(per); qa = np.sort(per)[min(m - 1, int(np.ceil((m + 1) * 0.9)) - 1)]
        zt = np.abs(Et) / (cs * st)
        cov_pix = float((zt <= q).mean())
        cov_seq = (zt <= q).reshape(len(Et), -1).mean(1)
        cov_acq = (zt <= qa).reshape(len(Et), -1).mean(1)
        ece, lv, ob = metrics.calibration(Et.ravel(), (cs * st).ravel())
        ece_raw, _, ob_raw = metrics.calibration(Et.ravel(), st.ravel())
        ause, fr, cu, co = metrics.sparsification(Et.ravel(), st.ravel())
        rho = spearmanr(np.abs(Et).ravel(), st.ravel()).correlation
        out[name] = dict(scale=float(cs), q=float(q), qa=float(qa), cov_pixel=cov_pix,
                         cov_seq_mean=float(cov_seq.mean()), cov_seq_min=float(cov_seq.min()),
                         frac_seq_ge90_acq=float((cov_acq >= 0.9).mean()), ece=ece, ece_raw=ece_raw, ause=ause,
                         spearman=float(rho), width=float(np.mean(2 * q * cs * st)), levels=lv.tolist(),
                         obs=ob.tolist(), obs_raw=ob_raw.tolist(), sp_fr=fr.tolist(), sp_unc=cu.tolist(),
                         sp_orc=co.tolist())
    json.dump(out, open(os.path.join(RES, 'uncertainty.json'), 'w'), indent=1)
    return out


def fig_uncertainty(u, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.25))
    lv = np.array(u['head']['levels'])
    ax[0].plot([0, 1], [0, 1], color='0.6', lw=0.8, ls=':', label='ideal')
    ax[0].plot(lv, u['head']['obs_raw'], color='0.55', ls='--', marker='o', ms=2.5, lw=1, label='head, raw')
    ax[0].plot(lv, u['head']['obs'], color='0.0', marker='s', ms=2.5, lw=1.3, label='head, scaled')
    ax[0].set_xlabel('Nominal coverage'); ax[0].set_ylabel('Observed coverage'); ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title('(a) Calibration', fontsize=7.5)
    for nm, ls, c in [('head', '-', '0.0'), ('ensemble', '-.', '0.35')]:
        ax[1].plot(u[nm]['sp_fr'], u[nm]['sp_unc'], ls=ls, color=c, lw=1.2, label=f'{"head" if nm == "head" else "ensemble"}')
    ax[1].plot(u['head']['sp_fr'], u['head']['sp_orc'], color='0.6', ls=':', lw=1, label='oracle')
    ax[1].set_xlabel('Fraction of pixels removed'); ax[1].set_ylabel('Normalised RMSE')
    ax[1].legend(fontsize=6, frameon=False); ax[1].set_title('(b) Sparsification', fontsize=7.5)
    Et, St = errors('test', SEEDS[0])
    s = (u['head']['scale'] * St).ravel(); e = np.abs(Et).ravel()
    bins = np.quantile(s, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
    ms = [s[idx == b].mean() for b in range(10)]; me = [np.sqrt((e[idx == b] ** 2).mean()) for b in range(10)]
    ax[2].plot([0, max(ms + me)], [0, max(ms + me)], color='0.6', ls=':', lw=0.8)
    ax[2].plot(ms, me, color='0.0', marker='o', ms=3, lw=1.2)
    ax[2].set_xlabel(r'Predicted $\hat{\upsilon}$ (dB, decile mean)'); ax[2].set_ylabel('Observed RMSE (dB)')
    ax[2].set_title('(c) Error vs. predicted spread', fontsize=7.5)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path, dpi=400); plt.close(fig)


def stages(n=32):
    """Stage-wise metrics of the first full-model seed on test sequences."""
    D = data.load('test')
    ck = torch.load(f'runs/full_s{SEEDS[0]}.pt', weights_only=False)
    net = train.build_model('full'); net.load_state_dict(ck['state']); net.eval()
    rows = []
    for i0 in range(0, n, 4):
        y, u0, psf = evaluate.tensors(D, i0, i0 + 4)
        with torch.no_grad():
            o = net(y, psf_true=psf, u0=u0, return_all=True)
        for j in range(4):
            i = i0 + j
            d = dict(pd_gt=D['pd_gt'][i], vessel=D['vessel'][i], micro=D['micro'][i], slow=D['slow'][i])
            u = D['u'][i].astype(np.float32)
            r = [dict(stage=0, epe=metrics.motion_epe(D['u0'][i].astype(np.float32), u))]
            for k, h in enumerate(o['hist']):
                m = metrics.image_metrics(h[j, 0].numpy(), d)
                m['epe'] = metrics.motion_epe(o['uhist'][k][j].numpy(), u); m['stage'] = k + 1
                r.append(m)
            rows.append(r)
    K = len(rows[0]) - 1
    agg = {key: [float(np.nanmean([rw[k][key] for rw in rows])) for k in range(1, K + 1)] for key in ['nrmse', 'ssim', 'auc', 'recall', 'epe']}
    agg['epe0'] = float(np.mean([rw[0]['epe'] for rw in rows]))
    # learned parameters
    sd = ck['state']
    agg['sv_th'] = torch.exp(sd['sv_th'])[:, :8].tolist()
    agg['beta'] = torch.sigmoid(sd['beta_l']).tolist()
    agg['eta'] = torch.exp(sd['log_eta']).tolist()
    agg['aL'] = torch.exp(sd['log_aL']).tolist(); agg['aS'] = torch.exp(sd['log_aS']).tolist()
    agg['lam'] = torch.exp(sd['log_lam']).tolist()
    agg['psf_a'] = sd['psf_a'].tolist(); agg['psf_b'] = sd['psf_b'].tolist()
    json.dump(agg, open(os.path.join(RES, 'stages.json'), 'w'), indent=1)
    return agg


def psf_accuracy():
    out = {}
    for reg in ['test', 'psfwide']:
        D = data.load(reg)
        errs = []
        for s in SEEDS:
            ex = np.load(os.path.join(RES, f'extra_{reg}_full@{s}.npz'))
            errs.append(np.abs(ex['sigma'] - D['psf']).mean(0))
        y = torch.tensor(D['y']); y = y / y.abs().pow(2).mean((1, 2, 3), keepdim=True).sqrt()
        import ops
        sz, sx = ops.psf_autocorr(y)
        raw = np.abs(np.stack([sz.numpy(), sx.numpy()], 1) - D['psf']).mean(0)
        out[reg] = dict(mae_corrected=np.mean(errs, 0).tolist(), mae_raw=raw.tolist(),
                        true_mean=D['psf'].mean(0).tolist())
    json.dump(out, open(os.path.join(RES, 'psf.json'), 'w'), indent=1)
    return out


def fig_stages(agg, path):
    D = data.load('test')
    ck = torch.load(f'runs/full_s{SEEDS[0]}.pt', weights_only=False)
    net = train.build_model('full'); net.load_state_dict(ck['state']); net.eval()
    i = 17
    y, u0, psf = evaluate.tensors(D, i, i + 1)
    with torch.no_grad():
        o = net(y, psf_true=psf, u0=u0, return_all=True)
    K = len(o['hist'])
    fig = plt.figure(figsize=(7.1, 3.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.38)
    gs = outer[0].subgridspec(1, K + 2, wspace=0.12)
    gb = outer[1].subgridspec(1, 3, wspace=0.62)
    for k in range(K):
        ax = fig.add_subplot(gs[0, k])
        h = o['hist'][k][0, 0].numpy()
        ax.imshow(10 * np.log10(h / np.percentile(h, 99.5) + 1e-12), cmap='gray', vmin=-40, vmax=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'Stage {k + 1}', fontsize=7)
    ax = fig.add_subplot(gs[0, K])
    ax.imshow(10 * np.log10(D['pd_gt'][i] / D['pd_gt'][i].max() + 1e-12), cmap='gray', vmin=-40, vmax=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title('Reference', fontsize=7)
    ax = fig.add_subplot(gs[0, K + 1])
    lam = o['lam'][0, 0].numpy()
    ax.imshow(np.log10(lam / np.median(lam)), cmap='gray_r')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(r'$\log_{10}\lambda_K/\mathrm{med}$', fontsize=7)
    st = np.arange(1, K + 1)
    a1 = fig.add_subplot(gb[0, 0])
    a1.plot(st, agg['auc'], 'k-o', ms=3, lw=1.2, label='AUC')
    a1.plot(st, agg['ssim'], color='0.45', ls='--', marker='s', ms=3, lw=1.1, label='SSIM')
    a1.plot(st, agg['recall'], color='0.3', ls=':', marker='^', ms=3, lw=1.1, label='Microvessel recall')
    a1.set_xlabel('Stage'); a1.set_ylim(0.25, 0.97); a1.legend(fontsize=6, frameon=False, loc='center right'); a1.grid(alpha=0.25, lw=0.5)
    a1.set_title('(a) Quality per stage', fontsize=7.5)
    a2 = fig.add_subplot(gb[0, 1])
    a2.plot([0] + list(st), [agg['epe0']] + agg['epe'], 'k-o', ms=3, lw=1.2)
    a2.set_xlabel('Stage (0: coarse registration)'); a2.set_xticks(range(0, K + 1)); a2.set_ylabel('EPE (px)'); a2.grid(alpha=0.25, lw=0.5)
    a2.set_title('(b) Displacement error', fontsize=7.5)
    a3 = fig.add_subplot(gb[0, 2])
    th = np.array(agg['sv_th'])
    for k in range(th.shape[0]):
        a3.plot(np.arange(1, th.shape[1] + 1), th[k], color=str(0.75 - 0.75 * k / max(1, th.shape[0] - 1)), lw=1,
                marker='.', ms=3)
    a3.set_xticks(range(1, th.shape[1] + 1)); a3.set_yscale('log'); a3.set_xlabel('Singular index $i$'); a3.set_ylabel(r'$\theta_{k,i}$')
    a3.set_title('(c) Spectral thresholds', fontsize=7.5); a3.grid(alpha=0.25, lw=0.5)
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.03); plt.close(fig)


if __name__ == '__main__':
    import sys
    what = sys.argv[1:]
    if 'unc' in what:
        u = uncertainty(); fig_uncertainty(u, 'figs/fig6_uncertainty.png')
        print({k: {kk: v for kk, v in d.items() if not isinstance(v, list)} for k, d in u.items()})
    if 'stages' in what:
        a = stages(); fig_stages(a, 'figs/fig5_stages.png'); print({k: v for k, v in a.items()})
    if 'psf' in what:
        print(psf_accuracy())
