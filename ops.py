"""Differentiable physics operators: Gaussian PSF (FFT), cubic backward warp and its exact adjoint,
axial phase modulation, noise-map and PSF estimators, motion update, utilities."""
import math
import torch
import torch.nn.functional as F

KAPPA = math.pi
SIGMA0 = 0.5


# ----------------------------------------------------------------------------- PSF
def gauss_tf(H, W, sz, sx, device=None):
    """Transfer function of a unit-sum Gaussian. sz, sx: tensors (B,) -> (B,1,H,W)."""
    fz = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
    fx = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)
    sz = sz.view(-1, 1, 1, 1); sx = sx.view(-1, 1, 1, 1)
    return torch.exp(-2 * math.pi ** 2 * ((sz * fz) ** 2 + (sx * fx) ** 2))


def blur(x, tf):
    """x: (B,T,H,W) complex; tf: (B,1,H,W) real. Self-adjoint (symmetric kernel)."""
    return torch.fft.ifft2(torch.fft.fft2(x) * tf)


# ----------------------------------------------------------------------------- warp
def _bspline_prefilter(fr):
    """Cubic B-spline coefficients via spectral division (fr: (N,2,H,W) real)."""
    H, W = fr.shape[-2:]
    wz = 2 * math.pi * torch.fft.fftfreq(H, device=fr.device).view(1, 1, H, 1)
    wx = 2 * math.pi * torch.fft.fftfreq(W, device=fr.device).view(1, 1, 1, W)
    tf = ((4 + 2 * torch.cos(wz)) / 6) * ((4 + 2 * torch.cos(wx)) / 6)
    return torch.fft.ifft2(torch.fft.fft2(fr) / tf).real


class Warp:
    """Backward warp f(x - u(x)) with cubic B-spline interpolation, u: (B,T,2,H,W) in pixels.
    Implemented as spectral prefiltering followed by four bilinear look-ups (positive B-spline
    weights allow pairwise merging). The exact adjoint is the vector-Jacobian product of this
    linear-in-f map."""

    PAD = 6

    def __init__(self, u):
        B, T, _, H, W = u.shape
        self.shape = (B, T, H, W)
        zz = torch.arange(H, device=u.device, dtype=u.dtype).view(1, 1, H, 1)
        xx = torch.arange(W, device=u.device, dtype=u.dtype).view(1, 1, 1, W)
        pz = zz - u[:, :, 0]; px = xx - u[:, :, 1]

        def taps(p):
            i = torch.floor(p).detach(); t = p - i
            w0 = (1 - t) ** 3 / 6; w1 = (4 - 6 * t ** 2 + 3 * t ** 3) / 6
            w2 = (1 + 3 * t + 3 * t ** 2 - 3 * t ** 3) / 6; w3 = t ** 3 / 6
            g0 = w0 + w1; g1 = w2 + w3
            return (g0, i - 1 + w1 / g0), (g1, i + 1 + w3 / g1)
        tz = taps(pz); tx = taps(px)
        self.terms = []
        for gz, hz in tz:
            for gx, hx in tx:
                Pd = self.PAD
                grid = torch.stack([2 * (hx + Pd) / (W + 2 * Pd - 1) - 1, 2 * (hz + Pd) / (H + 2 * Pd - 1) - 1], -1).reshape(B * T, H, W, 2)
                self.terms.append(((gz * gx).reshape(B * T, 1, H, W), grid))

    def _apply(self, fr):
        c = _bspline_prefilter(F.pad(fr, (self.PAD,) * 4, mode='reflect'))
        out = 0
        for g, grid in self.terms:
            out = out + g * F.grid_sample(c, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out

    def fwd(self, f):
        B, T, H, W = self.shape
        fr = torch.view_as_real(f.reshape(B * T, H, W)).permute(0, 3, 1, 2)
        o = self._apply(fr).permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(o).view(B, T, H, W)

    def adj(self, r):
        B, T, H, W = self.shape
        rr = torch.view_as_real(r.reshape(B * T, H, W)).permute(0, 3, 1, 2)
        need = torch.is_grad_enabled() and (rr.requires_grad or self.terms[0][1].requires_grad)
        with torch.enable_grad():
            f0 = torch.zeros(B * T, 2, H, W, device=r.device, requires_grad=True)
            o = self._apply(f0)
            g, = torch.autograd.grad(o, f0, grad_outputs=rr, create_graph=need)
        return torch.view_as_complex(g.permute(0, 2, 3, 1).contiguous()).view(B, T, H, W)


def phase(u):
    return torch.exp(1j * KAPPA * u[:, :, 0])


# ----------------------------------------------------------------------------- estimators
def smooth2d(x, sigma):
    """Real Gaussian smoothing over last two dims of (B,C,H,W) via FFT; sigma tensor or float."""
    H, W = x.shape[-2:]
    s = sigma if torch.is_tensor(sigma) else torch.tensor([float(sigma)], device=x.device)
    tf = gauss_tf(H, W, s.reshape(-1)[:1], s.reshape(-1)[:1], x.device)   # (1,1,H,W)
    if x.dim() == 5:
        tf = tf.unsqueeze(1)
    out = torch.fft.ifft2(torch.fft.fft2(x) * tf)
    return out if x.is_complex() else out.real


def noise_map(y, smooth=3.0):
    """Heteroscedastic noise power from the doubly-high-pass (Nyquist) component of the IQ frames."""
    k1 = torch.tensor([1., -6., 15., -20., 15., -6., 1.], device=y.device)   # 6th-order difference
    k = (k1[:, None] * k1[None, :]).view(1, 1, 7, 7)
    B, T, H, W = y.shape
    yr = torch.view_as_real(y).permute(0, 1, 4, 2, 3).reshape(B * T * 2, 1, H, W)
    hp = F.pad(F.conv2d(yr, k), (3, 3, 3, 3), mode='replicate')
    e = hp.view(B, T, 2, H, W).pow(2).sum(2).mean(1, keepdim=True) / k.pow(2).sum()
    return smooth2d(e, smooth).clamp_min(1e-12)       # (B,1,H,W)


def psf_autocorr(y, noise=None):
    """Closed-form Gaussian PSF width from lag-1 speckle autocorrelation (per sequence)."""
    def rho(a, b):
        return (a * b.conj()).sum((1, 2, 3)).abs() / (0.5 * (a.abs() ** 2 + b.abs() ** 2)).sum((1, 2, 3))
    y = y - y.mean(1, keepdim=True) * 0     # keep tissue speckle (dominant) as PSF probe
    rz = rho(y[:, :, 1:, :], y[:, :, :-1, :])
    rx = rho(y[:, :, :, 1:], y[:, :, :, :-1])
    if noise is not None:  # remove the white-noise contribution to zero-lag power
        P = (y.abs() ** 2).mean((1, 2, 3)); n = noise.mean((1, 2, 3))
        rz = rz * P / (P - n).clamp_min(1e-9); rx = rx * P / (P - n).clamp_min(1e-9)
    rz = rz.clamp(1e-3, 0.999); rx = rx.clamp(1e-3, 0.999)
    sz = torch.sqrt(-1.0 / (4 * torch.log(rz))); sx = torch.sqrt(-1.0 / (4 * torch.log(rx)))
    return sz, sx


def _dz(G):
    out = torch.zeros_like(G)
    out[..., 1:-1, :] = 0.5 * (G[..., 2:, :] - G[..., :-2, :])
    out[..., 0, :] = G[..., 1, :] - G[..., 0, :]; out[..., -1, :] = G[..., -1, :] - G[..., -2, :]
    return out


def _dx(G):
    out = torch.zeros_like(G)
    out[..., 1:-1] = 0.5 * (G[..., 2:] - G[..., :-2])
    out[..., 0] = G[..., 1] - G[..., 0]; out[..., -1] = G[..., -1] - G[..., -2]
    return out


def motion_update(y, G, u, omega, win, eta, eps=1e-2, use_phase=True):
    """Windowed Gauss-Newton refinement of the displacement on the complex IQ signal.
    Model m = exp(j*kappa*u_z) * G with G = W_u(h*L); perturbation delta gives
    dm ~= P*(j*kappa*G - dG/dz) delta_z - P*dG/dx delta_x. The 2x2 normal equations are
    accumulated with a Gaussian window of width `win` and solved per pixel and frame."""
    if use_phase:
        P = phase(u)
        m = P * G
        r = y - m
        Jz = P * (1j * KAPPA * G - _dz(G))
        Jx = -P * _dx(G)
    else:   # envelope-only (phase-free) step: robust to axial phase wrapping, used coarse-to-fine
        eg = smooth2d(G.abs().unsqueeze(2), 1.0).squeeze(2); ey = smooth2d(y.abs().unsqueeze(2), 1.0).squeeze(2)
        r = (ey - eg).to(torch.complex64)
        Jz = -_dz(eg).to(torch.complex64); Jx = -_dx(eg).to(torch.complex64)
    om = omega
    a11 = smooth2d(om * Jz.abs().pow(2), win); a22 = smooth2d(om * Jx.abs().pow(2), win)
    a12 = smooth2d(om * (Jz.conj() * Jx).real, win)
    b1 = smooth2d(om * (Jz.conj() * r).real, win); b2 = smooth2d(om * (Jx.conj() * r).real, win)
    reg = eps * (a11 + a22).mean((2, 3), keepdim=True) + 1e-12
    a11 = a11 + reg; a22 = a22 + reg
    det = a11 * a22 - a12 * a12
    dz = (a22 * b1 - a12 * b2) / det
    dx = (a11 * b2 - a12 * b1) / det
    du = torch.stack([dz, dx], 2).clamp(-0.5, 0.5)
    return u + eta * du


def regularize_motion(u, sigma, sigma_t=None):
    """Spatial Gaussian smoothing, optional temporal smoothing, zero temporal mean."""
    B, T, C, H, W = u.shape
    us = smooth2d(u.reshape(B, T * C, H, W), sigma).reshape(B, T, C, H, W)
    if sigma_t is not None:
        st = sigma_t.reshape(-1)[:1] if torch.is_tensor(sigma_t) else torch.tensor([float(sigma_t)])
        tt = torch.arange(-3, 4, dtype=u.dtype, device=u.device)
        k = torch.exp(-0.5 * (tt / st) ** 2); k = k / k.sum()
        up = torch.cat([us[:, :1].expand(B, 3, C, H, W), us, us[:, -1:].expand(B, 3, C, H, W)], 1)
        us = sum(k[i] * up[:, i:i + T] for i in range(7))
    return us - us.mean(1, keepdim=True)


def svt_rel(X, tau, weights=None):
    """Singular value soft-thresholding of the Casorati matrix (pixels x time) with threshold tau*s1.
    Eigen-decomposition of the T x T Gram matrix; eigenvectors are treated as constants in the
    backward pass while eigenvalues keep their exact first-order derivative."""
    B, T, H, W = X.shape
    C = X.reshape(B, T, H * W)                       # rows: time
    G = C @ C.conj().transpose(1, 2)                 # (B,T,T)
    with torch.no_grad():
        ev, V = torch.linalg.eigh(G.to(torch.complex128))
        V = V.to(C.dtype)
    lam = torch.einsum('bit,bij,bjt->bt', V.conj(), G, V).real.clamp_min(1e-12)   # eigenvalues
    s = torch.sqrt(lam)
    s1 = s.max(1, keepdim=True).values
    thr = tau * s1
    if weights is not None:
        thr = thr * weights.flip(0).view(1, -1)     # eigh is ascending: weights indexed from largest
    shrink = F.relu(1 - thr / s)
    P = V @ torch.diag_embed(shrink.to(C.dtype)) @ V.conj().transpose(1, 2)    # (B,T,T)
    return (P @ C).reshape(B, T, H, W)


def svt_spec(X, thr, smooth=False):
    """Learned spectral shrinkage: index-wise thresholds thr (T,) relative to the leading singular
    value, ordered from the largest singular value down."""
    B, T, H, W = X.shape
    C = X.reshape(B, T, H * W)
    G = C @ C.conj().transpose(1, 2)
    with torch.no_grad():
        ev, V = torch.linalg.eigh(G.to(torch.complex128))
        V = V.to(C.dtype).flip(-1)                    # descending order
    lam = torch.einsum('bit,bij,bjt->bt', V.conj(), G, V).real.clamp_min(1e-12)
    s = torch.sqrt(lam)
    t = thr.view(1, -1) * s[:, :1]
    if smooth:   # softplus-smoothed singular value shrinkage (keeps gradients for every index)
        shrink = 0.05 * t * F.softplus((s - t) / (0.05 * t)) / s
    else:
        shrink = F.relu(1 - t / s)
    P = V @ torch.diag_embed(shrink.to(C.dtype)) @ V.conj().transpose(1, 2)
    return (P @ C).reshape(B, T, H, W)


def lowrank_project(X, k):
    """Hard rank-k approximation along time (used by non-learned baselines)."""
    B, T, H, W = X.shape
    C = X.reshape(B, T, H * W)
    G = C @ C.conj().transpose(1, 2)
    ev, V = torch.linalg.eigh(G.to(torch.complex128))
    V = V.to(C.dtype)[:, :, T - k:] if k > 0 else V.to(C.dtype)[:, :, :0]
    P = V @ V.conj().transpose(1, 2)
    return (P @ C).reshape(B, T, H, W)


def oriented_energy(E, n_orient=4, s_long=2.5, s_short=0.6):
    """Max over orientations of anisotropic Gaussian-smoothed energy (B,1,H,W)."""
    H, W = E.shape[-2:]
    fz = torch.fft.fftfreq(H, device=E.device).view(1, 1, H, 1)
    fx = torch.fft.fftfreq(W, device=E.device).view(1, 1, 1, W)
    Ef = torch.fft.fft2(E)
    outs = []
    for o in range(n_orient):
        th = math.pi * o / n_orient
        fa = fz * math.cos(th) + fx * math.sin(th)
        fb = -fz * math.sin(th) + fx * math.cos(th)
        tf = torch.exp(-2 * math.pi ** 2 * ((s_long * fa) ** 2 + (s_short * fb) ** 2))
        outs.append(torch.fft.ifft2(Ef * tf).real)
    return torch.stack(outs, 0).max(0).values.clamp_min(0)


def vesselness(logpd):
    """Frangi-type bright-tube measure of a (B,1,H,W) log power map, normalised to [0,1]."""
    p = smooth2d(logpd, 1.0)
    pz = F.pad(p, (1, 1, 1, 1), mode='replicate')
    hzz = pz[:, :, 2:, 1:-1] - 2 * p + pz[:, :, :-2, 1:-1]
    hxx = pz[:, :, 1:-1, 2:] - 2 * p + pz[:, :, 1:-1, :-2]
    hzx = 0.25 * (pz[:, :, 2:, 2:] - pz[:, :, 2:, :-2] - pz[:, :, :-2, 2:] + pz[:, :, :-2, :-2])
    tr = hzz + hxx; dt = torch.sqrt(((hzz - hxx) / 2) ** 2 + hzx ** 2 + 1e-12)
    l1 = tr / 2 + dt; l2 = tr / 2 - dt          # l2 <= l1
    rb = (l1.abs() / (l2.abs() + 1e-6)) ** 2
    sn = l1 ** 2 + l2 ** 2
    c = sn.flatten(1).amax(1).view(-1, 1, 1, 1).sqrt() * 0.5 + 1e-6
    v = torch.exp(-rb / (2 * 0.5 ** 2)) * (1 - torch.exp(-sn / (2 * c ** 2))) * (l2 < 0).float()
    return v / (v.flatten(1).amax(1).view(-1, 1, 1, 1) + 1e-6)
