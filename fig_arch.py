"""Architecture diagrams (Fig. 1 overall pipeline, Fig. 2 single unrolled stage), grayscale."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif'})


def box(ax, x, y, w, h, text, fc='white', lw=1.0, fs=7.2, ls='-', bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.06', fc=fc, ec='black', lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal',
            linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, ls='-', lw=1.0, text=None, toff=(0, 0.08), fs=6.5, ha='center', rad=None):
    st = dict(arrowstyle='-|>', lw=lw, color='black', linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=9)
    if rad is not None:
        st['connectionstyle'] = f'arc3,rad={rad}'
    ax.annotate('', xy=p1, xytext=p0, arrowprops=st)
    if text:
        ax.text((p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1], text, fontsize=fs, ha=ha, va='bottom')


def poly(ax, pts, ls='-', lw=1.0, head=True):
    """Orthogonal poly-line with an arrow head on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=ls, solid_capstyle='butt')
    if head:
        arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw)
    else:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color='black', lw=lw, ls=ls)


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b, f=0.5):
    return (b[0] + f * b[2], b[1] + b[3])


def Bm(b, f=0.5):
    return (b[0] + f * b[2], b[1])


def fig_overall(path):
    W, H = 7.2, 3.95
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    G1, G2 = '#f2f2f2', '#d9d9d9'
    b_in = box(ax, 0.10, 1.75, 0.95, 0.80, 'Ultrafast IQ\nensemble\n$\\mathbf{Y}\\in\\mathbb{C}^{N\\times T}$', fc=G1)
    b_reg = box(ax, 1.35, 2.55, 1.45, 0.80, 'Coarse-to-fine\nregistration\nenvelope $\\rightarrow$ complex GN', fs=6.8)
    b_psf = box(ax, 1.35, 0.85, 1.45, 0.80, 'Speckle-autocorrelation\nPSF estimate\n$\\hat{\\sigma}_z,\\ \\hat{\\sigma}_x$', fs=6.8)
    b_init = box(ax, 3.20, 1.70, 1.00, 0.90, 'Back-projection,\nrank-3 tissue\n$\\mathbf{L}^{(0)},\\ s_{\\mathrm{ref}}$', fs=6.8)
    b_s1 = box(ax, 4.50, 1.70, 0.75, 0.90, 'Stage 1', fc=G2, bold=True)
    ax.text(5.45, 2.15, r'$\cdots$', fontsize=12, ha='center', va='center')
    b_sk = box(ax, 5.65, 1.70, 0.75, 0.90, 'Stage $K$', fc=G2, bold=True)
    b_out = box(ax, 6.62, 1.40, 0.42, 1.50, 'Power\nDoppler\n$\\hat{P}$\n\nUncert.\n$\\hat{\\sigma}$\n\nMotion\n$\\hat{\\mathbf{u}}$', fc=G1, fs=6.2)
    b_par = box(ax, 4.20, 3.22, 2.45, 0.58, 'Learnable parameters $\\Theta$: step sizes, spectral\nthresholds, ParamNet, PSF correction,\nmotion gains, uncertainty head', fs=6.0)
    b_loss = box(ax, 4.30, 0.15, 2.15, 0.58, 'Training loss $\\mathcal{L}$: scale-invariant dB\n+ displacement + Gaussian NLL', fs=6.4)
    # forward arrows
    arrow(ax, (1.05, 2.35), (1.35, 2.95))
    arrow(ax, (1.05, 1.95), (1.35, 1.25))
    poly(ax, [R(b_reg), (3.00, 2.95), (3.00, 2.35), (3.20, 2.35)])
    ax.text(3.04, 2.62, r'$\mathbf{u}^{(0)}$', fontsize=7, ha='left', va='center')
    poly(ax, [R(b_psf), (3.00, 1.25), (3.00, 1.95), (3.20, 1.95)])
    ax.text(3.04, 1.52, r'$\hat{\sigma}$', fontsize=7, ha='left', va='center')
    arrow(ax, R(b_init), L(b_s1))
    arrow(ax, R(b_s1), (5.33, 2.15))
    arrow(ax, (5.57, 2.15), L(b_sk))
    arrow(ax, R(b_sk), L(b_out))
    # raw data enters every stage (data consistency)
    poly(ax, [Bm(b_in, 0.5), (0.575, 0.55), (3.90, 0.55), (3.90, 1.35), (4.875, 1.35), Bm(b_s1)])
    poly(ax, [(4.875, 1.35), (6.025, 1.35), Bm(b_sk)])
    ax.text(1.40, 0.59, r'$\mathbf{Y}$ (data consistency in every stage)', fontsize=6.2, ha='left', va='bottom')
    # parameters feed stages
    arrow(ax, (4.875, 3.22), T(b_s1))
    arrow(ax, (6.025, 3.22), T(b_sk))
    # outputs to loss
    poly(ax, [Bm(b_out), (6.83, 0.58), (6.45, 0.58)])
    # dashed reverse-mode gradient from the loss back to the learnable parameters
    poly(ax, [(6.45, 0.30), (7.12, 0.30), (7.12, 3.51), (6.65, 3.51)], ls='--')
    ax.text(7.08, 3.08, r'$\nabla_{\Theta}\mathcal{L}$', fontsize=6.6, ha='right', va='center')
    ax.text(0.10, 3.72, 'Non-learned front end', fontsize=6.6, style='italic')
    ax.text(0.10, 0.12, 'Solid: forward pass.  Dashed: reverse-mode gradient.', fontsize=6.2, style='italic')
    fig.savefig(path, dpi=400); plt.close(fig)


def fig_stage(path):
    W, H = 7.2, 3.95
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    G1, G2 = '#f2f2f2', '#d9d9d9'
    b_state = box(ax, 0.08, 1.55, 0.88, 0.95, 'State $k-1$\n$\\mathbf{L},\\ \\mathbf{S},\\ \\mathbf{u}$', fc=G1)
    b_res = box(ax, 1.25, 2.55, 1.35, 0.80, 'Residual\n$\\mathbf{R}=\\mathbf{Y}-\\mathcal{M}_{\\mathbf{u}}(\\mathbf{L}+h\\ast\\mathbf{S})$', fs=6.6)
    b_w = box(ax, 1.25, 1.40, 1.35, 0.72, 'Reliability weights\n$\\Omega$ (re-weighted)', fs=6.8)
    b_tis = box(ax, 2.95, 2.55, 1.35, 0.80, 'Tissue step +\nspectral shrinkage\n$\\mathcal{P}_{\\boldsymbol{\\theta}_k}$', fc=G2, fs=6.8)
    b_bl = box(ax, 2.95, 1.30, 1.35, 0.92, 'Blood step ($h^{\\top}$) +\nvessel-preserving\nstructured shrinkage', fc=G2, fs=6.8)
    b_pn = box(ax, 2.95, 0.12, 1.35, 0.78, 'ParamNet: $\\lambda_k(\\mathbf{x})$\nfrom reliability,\nmotion, vesselness', fc=G2, fs=6.6)
    b_or = box(ax, 4.65, 1.18, 1.15, 0.78, 'Oriented group\nenergy $\\tilde{E}(\\mathbf{x})$', fs=6.8)
    b_mo = box(ax, 4.65, 2.55, 1.15, 0.80, 'Motion increment\n(complex GN,\nregularised)', fc=G2, fs=6.8)
    b_next = box(ax, 6.15, 1.55, 0.95, 0.95, 'State $k$\n$\\mathbf{L},\\ \\mathbf{S},\\ \\mathbf{u}$', fc=G1)
    # state -> residual
    poly(ax, [(0.96, 2.25), (1.10, 2.25), (1.10, 2.95), L(b_res)])
    arrow(ax, Bm(b_res), T(b_w))
    arrow(ax, R(b_res), L(b_tis))
    # weights -> blood and tissue
    arrow(ax, (2.60, 1.76), (2.95, 1.76))
    poly(ax, [(2.60, 1.95), (2.75, 1.95), (2.75, 2.75), (2.95, 2.75)])
    # Gauss-Seidel residual update tissue -> blood
    arrow(ax, Bm(b_tis, 0.5), T(b_bl, 0.5))
    ax.text(3.68, 2.38, 'updated $\\mathbf{R}$', fontsize=6.2, ha='left', va='center')
    # ParamNet -> blood
    arrow(ax, T(b_pn), Bm(b_bl))
    # blood <-> oriented energy
    arrow(ax, (4.30, 1.78), (4.65, 1.78))
    arrow(ax, (4.65, 1.40), (4.30, 1.40))
    # tissue -> motion (static reference) and blood -> motion (blood removal)
    arrow(ax, R(b_tis), L(b_mo))
    ax.text(4.47, 2.99, r'$\bar{\mathbf{L}}$', fontsize=6.6, ha='center', va='bottom')
    poly(ax, [(4.30, 2.12), (5.00, 2.12), (5.00, 2.55)])
    ax.text(5.05, 2.30, r'$\mathbf{S}$', fontsize=6.6, ha='left', va='center')
    # outputs to the next state
    poly(ax, [T(b_tis, 0.5), (3.625, 3.62), (6.62, 3.62), T(b_next, 0.5)])
    ax.text(5.15, 3.66, r'$\mathbf{L}$', fontsize=6.6, ha='center', va='bottom')
    poly(ax, [R(b_mo), (5.98, 2.95), (5.98, 2.30), (6.15, 2.30)])
    ax.text(6.02, 2.62, r'$\mathbf{u}$', fontsize=6.6, ha='left', va='center')
    poly(ax, [(4.10, 1.30), (4.10, 1.02), (5.98, 1.02), (5.98, 1.76), (6.15, 1.76)])
    ax.text(5.20, 0.98, r'$\mathbf{S}$', fontsize=6.6, ha='center', va='top')
    # state also feeds ParamNet features (motion magnitude etc.)
    poly(ax, [Bm(b_state), (0.52, 0.51), L(b_pn)])
    # dashed reverse-mode gradient into the stage parameters
    poly(ax, [(6.62, 1.55), (6.62, 0.51), R(b_pn)], ls='--')
    ax.text(6.66, 0.80, r'$\nabla_{\Theta_k}\mathcal{L}$', fontsize=6.6, ha='left', va='center')
    ax.text(0.08, 3.75, 'Unrolled stage $k$ (blocks holding parameters $\\Theta_k$ shaded)', fontsize=6.8, style='italic')
    fig.savefig(path, dpi=400); plt.close(fig)


if __name__ == '__main__':
    import os
    os.makedirs('figs', exist_ok=True)
    fig_overall('figs/fig1_overall.png')
    fig_stage('figs/fig2_stage.png')
