import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'font.size': 8})
regs = [('test', 'Nominal'), ('psfwide', 'Wide\nPSF'), ('tubes', 'Phantom-\nlike'), ('penetrating', 'Pulsatile\ncortex'),
        ('slowflow', 'Very slow\nflow'), ('decor', 'Tissue\ndecorr.')]
ms = ['svd', 'mcsvd', 'jpa', 'durpca', 'cnn', 'full']
fills = ['1.0', '0.85', '0.65', '0.45', '0.3', '0.0']; hatches = ['', '///', '', 'xx', '...', '']
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.28))
w = 0.13
for j, m in enumerate(ms):
    vals, lo, hi = [], [], []
    for r, _ in regs:
        v, _ = analysis.per_seq(analysis.load(r), m, 'auc'); c = analysis.boot_ci(v)
        vals.append(np.nanmean(v)); lo.append(np.nanmean(v) - c[0]); hi.append(c[1] - np.nanmean(v))
    x = np.arange(len(regs)) + (j - 2.5) * w
    ax[0].bar(x, vals, w, color=fills[j], edgecolor='black', lw=0.5, hatch=hatches[j], label=analysis.NAMES[m],
              yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.2))
ax[0].set_xticks(np.arange(len(regs))); ax[0].set_xticklabels([n for _, n in regs], fontsize=6.8)
ax[0].set_ylim(0.35, 1.0); ax[0].set_ylabel('AUC'); ax[0].grid(axis='y', alpha=0.25, lw=0.5)
ax[0].legend(ncol=6, fontsize=6.2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.6)
ax[0].set_title('(a) Generalisation and failure regimes', fontsize=7.5, pad=4)
tm = json.load(open('results/timing.json')); reg = tm['registration']
res = analysis.load('test')
for m, mk in zip(ms + ['asvd', 'rpca'], ['o', 's', 'D', '<', '>', '*', 'v', '^']):
    v, _ = analysis.per_seq(res, m, 'auc')
    t = tm[m] + (reg if m in ('mcsvd', 'jpa', 'full') else 0)
    ax[1].scatter(t, np.nanmean(v), marker=mk, s=40 if m == 'full' else 22, color='black' if m == 'full' else '0.45', zorder=3)
    dx = 1.25
    off = {'svd': (6, -9), 'asvd': (6, 3), 'full': (-38, -9), 'jpa': (-12, -11), 'durpca': (-10, 6), 'cnn': (-24, -10), 'mcsvd': (-30, -10), 'rpca': (5, -2)}[m]
    ax[1].annotate(analysis.NAMES[m], (t, np.nanmean(v)), xytext=off, textcoords='offset points', fontsize=6.2)
ax[1].set_xscale('log'); ax[1].set_xlabel('Time per ensemble (s)'); ax[1].set_ylabel('AUC (nominal test)')
ax[1].set_xlim(2e-3, 30); ax[1].grid(alpha=0.25, lw=0.5); ax[1].set_title('(b) Accuracy versus run time', fontsize=7.5, pad=4)
fig.savefig('figs/fig7_shift_eff.png', dpi=400, bbox_inches='tight', pad_inches=0.03)
