"""Image-quality, vascular, motion and calibration metrics for power-Doppler reconstructions."""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

DR = 40.0  # dB display / evaluation dynamic range


def to_db01(pd, ref=None):
    pd = np.maximum(pd, 0)
    ref = np.percentile(pd, 99.5) if ref is None else ref
    db = 10 * np.log10(pd / (ref + 1e-30) + 1e-12)
    return np.clip(db, -DR, 0) / DR + 1.0


def ssim(a, b, sigma=1.5, L=1.0):
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a = gaussian_filter(a, sigma); mu_b = gaussian_filter(b, sigma)
    saa = gaussian_filter(a * a, sigma) - mu_a ** 2
    sbb = gaussian_filter(b * b, sigma) - mu_b ** 2
    sab = gaussian_filter(a * b, sigma) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(m.mean())


def auc_score(pos, neg):
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


M = 4  # evaluation margin (pixels) excluded on every side for all methods


def _crop(d, pd):
    c = (slice(M, -M), slice(M, -M))
    return pd[c], dict(pd_gt=d['pd_gt'][c], vessel=d['vessel'][c], micro=d['micro'][c], slow=d['slow'][c])


def image_metrics(pd, d):
    """pd: estimated power map (H,W); d: simulation dict. Evaluated inside the margin-cropped FOV."""
    pd, d = _crop(d, pd)
    gt = d['pd_gt']
    a = to_db01(pd); b = to_db01(gt, ref=gt.max())
    nrmse = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    psnr = float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))
    v = d['vessel']; bg = ~binary_dilation(v, iterations=1)
    mv, mb, sb = pd[v].mean(), pd[bg].mean(), pd[bg].std() + 1e-30
    cr = float(10 * np.log10(mv / (mb + 1e-30) + 1e-30))
    vsnr = float(20 * np.log10(abs(mv - mb) / sb + 1e-12))
    thr = np.percentile(pd[bg], 95)
    micro = d['micro']
    rec = float((pd[micro] > thr).mean()) if micro.any() else np.nan
    slow = d['slow']
    rec_slow = float((pd[slow] > thr).mean()) if slow.any() else np.nan
    auc = auc_score(pd[v], pd[bg])
    return dict(nrmse=nrmse, psnr=psnr, ssim=ssim(a, b), cr=cr, vsnr=vsnr, recall=rec,
                recall_slow=rec_slow, auc=auc)


def motion_epe(u_hat, u):
    if u_hat is None:
        return np.nan
    return float(np.sqrt(((u_hat - u) ** 2).sum(1))[:, M:-M, M:-M].mean())


def db_error(pd, d):
    """Signed error of the normalised dB map (used for uncertainty calibration)."""
    pd, d = _crop(d, pd)
    gt = d['pd_gt']
    return (to_db01(pd) - to_db01(gt, ref=gt.max())) * DR


def calibration(err, std, levels=np.linspace(0.05, 0.95, 19)):
    """Regression calibration: observed vs nominal coverage of central Gaussian intervals."""
    from scipy.stats import norm
    z = np.abs(err) / (std + 1e-9)
    obs = np.array([(z <= norm.ppf(0.5 + p / 2)).mean() for p in levels])
    ece = float(np.abs(obs - levels).mean())
    return ece, levels, obs


def sparsification(err, std, n=20):
    """Area under the sparsification-error curve (oracle-normalised RMSE removal)."""
    e2 = err ** 2
    o_unc = np.argsort(-std); o_orc = np.argsort(-np.abs(err))
    fr = np.linspace(0, 0.95, n)
    N = len(e2)
    cu = [np.sqrt(e2[o_unc[int(f * N):]].mean()) for f in fr]
    co = [np.sqrt(e2[o_orc[int(f * N):]].mean()) for f in fr]
    cu, co = np.array(cu) / cu[0], np.array(co) / co[0]
    return float(np.trapezoid(cu - co, fr)), fr, cu, co
