"""In-silico ultrafast Doppler simulator (baseband IQ, compounded plane-wave frames).

Each sequence contains
  * tissue speckle with smooth echogenicity, partial out-of-plane decorrelation,
  * a vessel network (tortuous curves, diameters 1-4 px, parabolic flow profiles),
  * rigid + non-rigid + strain tissue motion (zero temporal mean),
  * axial-motion phase modulation exp(j*kappa*u_z),
  * a Gaussian PSF of unknown, sequence-specific width,
  * depth-dependent heteroscedastic complex noise.
Ground truth: blood power map at the target resolution (sigma0), blood IQ at target
resolution in reference coordinates, displacement field and vessel masks.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

KAPPA = np.pi          # rad of Doppler phase per pixel of axial displacement (dz = lambda/4)
SIGMA0 = 0.5           # target PSF width (pixels)

DEFAULT = dict(H=64, W=64, T=32,
               ctr_db=(25., 40.),        # clutter-to-blood power ratio
               bnr_db=(10., 20.),         # blood (large vessel) to noise ratio at top of image
               depth_gain=(1.5, 3.0),    # noise std multiplier at bottom
               rigid_amp=(0.0, 1.2), nonrigid_amp=(0.0, 0.7), strain=(0.0, 0.015),
               motion_cycles=(0.3, 1.5), hf_motion=0.0,
               decor=0.0,        # tissue out-of-plane decorrelation per frame
               sig_z=(1.2, 1.9), sig_x=(1.4, 2.6),
               n_vessels=(5, 10), geometry='tortuous', speed=(0.03, 0.45))


def _u(rng, r):
    return rng.uniform(*r) if isinstance(r, (tuple, list)) else r


def gaussian_kernel_fft(H, W, sz, sx):
    """Unit-sum Gaussian PSF transfer function (real, centred)."""
    fz = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.exp(-2 * np.pi ** 2 * ((sz * fz) ** 2 + (sx * fx) ** 2))


def blur(img, sz, sx):
    Hf = gaussian_kernel_fft(img.shape[-2], img.shape[-1], sz, sx)
    return np.fft.ifft2(np.fft.fft2(img, axes=(-2, -1)) * Hf, axes=(-2, -1))


def _curve(rng, H, W, geometry):
    if geometry == 'tubes':         # straight, mostly lateral tubes (flow-phantom-like)
        z = rng.uniform(6, H - 6)
        ang = rng.uniform(-0.35, 0.35)
        t = np.linspace(0, 1, 200)
        xs = -5 + t * (W + 10)
        zs = z + np.tan(ang) * (xs - W / 2)
        return np.stack([zs, xs], 1)
    if geometry == 'penetrating':   # mostly axial, cortex-like
        x = rng.uniform(4, W - 4)
        t = np.linspace(0, 1, 200)
        zs = -3 + t * rng.uniform(0.4, 1.0) * (H + 6)
        xs = x + 2.5 * np.sin(2 * np.pi * t * rng.uniform(0.3, 1.2) + rng.uniform(0, 6.3))
        return np.stack([zs, xs], 1)
    # tortuous cubic Bezier through the field of view
    P = np.stack([rng.uniform(-8, H + 8, 4), rng.uniform(-8, W + 8, 4)], 1)
    P[0] = [rng.uniform(0, H), -6] if rng.random() < .5 else [-6, rng.uniform(0, W)]
    t = np.linspace(0, 1, 300)[:, None]
    B = ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
    return B


def vessel_network(rng, cfg):
    H, W = cfg['H'], cfg['W']
    amp = np.zeros((H, W)); vz = np.zeros((H, W)); vx = np.zeros((H, W))
    diam_map = np.zeros((H, W))
    nv = rng.integers(cfg['n_vessels'][0], cfg['n_vessels'][1] + 1)
    zz, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([zz.ravel(), xx.ravel()], 1).astype(float)
    for _ in range(nv):
        d = rng.choice([1.0, 1.5, 2.0, 3.0, 4.0], p=[.3, .25, .2, .15, .1])
        C = _curve(rng, H, W, cfg['geometry'])
        # distance of every pixel to the polyline (dense sampling suffices)
        from scipy.spatial import cKDTree
        tree = cKDTree(C)
        dist, idx = tree.query(pts)
        dist = dist.reshape(H, W); idx = idx.reshape(H, W)
        tang = np.gradient(C, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        r = d / 2
        inside = dist <= r
        if not inside.any():
            continue
        prof = np.clip(1 - (dist / max(r, 0.5)) ** 2, 0, 1)        # parabolic profile
        vmax = rng.uniform(*cfg['speed']) * (0.5 + 0.5 * d / 4)    # small vessels slower
        sgn = rng.choice([-1, 1])
        a = (0.35 + 0.65 * min(d, 3) / 3)                          # weaker echoes in small vessels
        sel = inside & (amp < a)
        amp[sel] = a
        diam_map[sel] = d
        v = sgn * vmax * np.maximum(prof, 0.15)
        vz[sel] = (v * tang[idx, 0])[sel]
        vx[sel] = (v * tang[idx, 1])[sel]
    return amp, vz, vx, diam_map


def motion_field(rng, cfg, T):
    H, W = cfg['H'], cfg['W']
    t = np.arange(T)
    cyc = _u(rng, cfg['motion_cycles'])
    ph = rng.uniform(0, 2 * np.pi, 4)
    A = _u(rng, cfg['rigid_amp']); Bn = _u(rng, cfg['nonrigid_amp']); eps = _u(rng, cfg['strain'])
    dirr = rng.normal(size=2); dirr /= np.linalg.norm(dirr)
    s1 = np.sin(2 * np.pi * cyc * t / T + ph[0])
    s2 = np.sin(2 * np.pi * cyc * t / T + ph[1])
    drift = np.linspace(-1, 1, T) * rng.uniform(-0.3, 0.3) * min(A, 1.0)
    u = np.zeros((T, 2, H, W))
    # rigid
    u[:, 0] += (A * dirr[0] * s1 + drift)[:, None, None]
    u[:, 1] += (A * dirr[1] * s1)[:, None, None]
    # smooth non-rigid field
    for c in range(2):
        f = gaussian_filter(rng.normal(size=(H, W)), 10, mode='wrap')
        f /= np.abs(f).max() + 1e-9
        u[:, c] += Bn * f[None] * s2[:, None, None]
    # axial strain (compression about mid-depth)
    zz = (np.arange(H) - H / 2)[None, :, None]
    u[:, 0] += eps * zz * np.sin(2 * np.pi * cyc * t / T + ph[2])[:, None, None]
    hf = cfg.get('hf_motion', 0.0)
    if hf > 0:   # pulsatile, non-smooth (out-of-distribution) component
        pulse = np.maximum(np.sin(2 * np.pi * 4 * t / T + ph[3]), 0) ** 4
        f = gaussian_filter(rng.normal(size=(H, W)), 4, mode='wrap'); f /= np.abs(f).max() + 1e-9
        u[:, 0] += hf * pulse[:, None, None] * f[None]
    u -= u.mean(0, keepdims=True)           # zero temporal mean: reference = mean configuration
    return u


def warp_np(f, u):
    """Backward warp f(x - u(x)) of a complex image f (H,W) with field u (2,H,W)."""
    H, W = f.shape
    zz, xx = np.mgrid[0:H, 0:W].astype(float)
    cz = np.clip(zz - u[0], 0, H - 1); cx = np.clip(xx - u[1], 0, W - 1)
    re = map_coordinates(f.real, [cz, cx], order=5, mode='nearest')
    im = map_coordinates(f.imag, [cz, cx], order=5, mode='nearest')
    return re + 1j * im


def simulate(seed, components=False, **over):
    cfg = dict(DEFAULT); cfg.update(over)
    rng = np.random.default_rng(seed)
    PAD = 8
    H0, W0 = cfg['H'], cfg['W']
    cfg['H'], cfg['W'] = H0 + 2 * PAD, W0 + 2 * PAD          # simulate a larger field, crop the FOV
    H, W, T = cfg['H'], cfg['W'], cfg['T']
    cz_, cx_ = slice(PAD, PAD + H0), slice(PAD, PAD + W0)
    sz, sx = _u(rng, cfg['sig_z']), _u(rng, cfg['sig_x'])
    amp, vz, vx, dmap = vessel_network(rng, cfg)
    # --- blood reflectivity process in reference coordinates (AR(1) speckle + Doppler phase)
    speed = np.sqrt(vz ** 2 + vx ** 2)
    rho = np.exp(-1.2 * speed / max(sx, sz))                  # decorrelation by transit through PSF
    rho = np.clip(rho, 0.2, 0.995)
    b = np.zeros((T, H, W), complex)
    w = (rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))) / np.sqrt(2)
    phase = np.zeros((H, W))
    for t in range(T):
        if t > 0:
            n = (rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))) / np.sqrt(2)
            w = rho * w + np.sqrt(1 - rho ** 2) * n
            phase = phase + KAPPA * vz
        b[t] = amp * w * np.exp(1j * phase)
    # --- tissue reflectivity (smooth echogenicity, partial decorrelation, no tissue in vessels)
    echo = 0.6 + 0.8 * (gaussian_filter(rng.uniform(size=(H, W)), 6) - 0.5) * 3
    echo = np.clip(echo, 0.25, 1.8) * (amp == 0) + 0.15 * (amp > 0)
    eps = _u(rng, cfg['decor'])
    r = echo * (rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))) / np.sqrt(2)
    tis = np.zeros((T, H, W), complex)
    for t in range(T):
        if t > 0 and eps > 0:
            n = echo * (rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))) / np.sqrt(2)
            r = np.sqrt(1 - eps) * r + np.sqrt(eps) * n
        tis[t] = r
    # scale tissue to the requested clutter-to-blood ratio (relative to mean blood power in vessels)
    pb = np.mean(np.abs(b[:, amp > 0]) ** 2) if (amp > 0).any() else 1.0
    ctr = _u(rng, cfg['ctr_db'])
    tis *= np.sqrt(pb * 10 ** (ctr / 10) / np.mean(np.abs(tis) ** 2))
    # --- motion, PSF, phase modulation
    u = motion_field(rng, cfg, T)
    y = np.zeros((T, H, W), complex); ytis = np.zeros((T, H, W), complex)
    pbb = 0.0
    for t in range(T):
        bb = blur(b[t], sz, sx)
        pbb += np.mean(np.abs(bb[amp > 0]) ** 2) / T if (amp > 0).any() else 1.0 / T
        ti = blur(tis[t], sz, sx)
        img = warp_np(ti + bb, u[t]) * np.exp(1j * KAPPA * u[t, 0])
        y[t] = img
        if components:
            ytis[t] = warp_np(ti, u[t]) * np.exp(1j * KAPPA * u[t, 0])
    # --- crop to the field of view (tissue/blood entering from outside the FOV is unknown to models)
    y = y[:, cz_, cx_]; ytis = ytis[:, cz_, cx_]
    tis0_full = blur(tis[0], sz, sx)
    H, W = H0, W0
    # --- heteroscedastic noise
    bnr = _u(rng, cfg['bnr_db'])
    g = _u(rng, cfg['depth_gain'])
    prof = 1 + (g - 1) * (np.arange(H) / (H - 1)) ** 1.5
    patch = 1 + 0.6 * np.clip(gaussian_filter(rng.normal(size=(H, W)), 8) * 8, 0, None)
    sig = np.sqrt(pbb / 10 ** (bnr / 10)) * prof[:, None] * patch
    y += sig[None] * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) / np.sqrt(2)
    # --- ground truth at target resolution, reference coordinates
    s0 = SIGMA0
    Hf, Wf = H0 + 2 * PAD, W0 + 2 * PAD
    b0 = np.stack([blur(b[t], s0, s0) for t in range(T)])[:, cz_, cx_]
    h0 = np.real(np.fft.ifft2(gaussian_kernel_fft(Hf, Wf, s0, s0)))
    pd_gt = np.real(np.fft.ifft2(np.fft.fft2(amp ** 2) * np.fft.fft2(h0 ** 2)))   # E|h0*b|^2
    pd_gt = np.maximum(pd_gt, 0)[cz_, cx_]
    amp = amp[cz_, cx_]; dmap = dmap[cz_, cx_]; vz = vz[cz_, cx_]; u = u[:, :, cz_, cx_]
    vessel = amp > 0
    micro = vessel & (dmap <= 2.0)
    slow = vessel & (np.abs(vz) < 0.08)
    return dict(y=y.astype(np.complex64), b0=b0.astype(np.complex64), pd_gt=pd_gt.astype(np.float32),
                u=u.astype(np.float32), vessel=vessel, micro=micro, slow=slow,
                noise_std=sig.astype(np.float32), psf=(sz, sx), ctr=ctr, bnr=bnr,
                ytis=ytis.astype(np.complex64) if components else None,
                tis0=tis0_full[cz_, cx_].astype(np.complex64) if components else None)


if __name__ == '__main__':
    import time
    t0 = time.time(); d = simulate(0); print('sim time', time.time() - t0)
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in d.items()})
    print('vessel frac', d['vessel'].mean(), 'micro', d['micro'].mean())
