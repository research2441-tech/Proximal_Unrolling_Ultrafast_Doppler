"""Physics-informed proximal unrolling for joint tissue / blood / displacement reconstruction,
plus the learned baselines (deep-unfolded RPCA and a direct CNN)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops

FULL = dict(adaptive=True, structured=True, group=True, psf='est', motion=True, weights=True,
            phase_model=True, uncertainty=True)


def _ifs(x):
    return math.log(math.expm1(x))


class ParamNet(nn.Module):
    """Maps local reliability / motion / vascular evidence to a log-multiplier of the sparsity weight."""

    def __init__(self, cin=7, width=24):
        super().__init__()
        self.c1 = nn.Conv2d(cin, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(width, width, 3, padding=1)
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, f):
        h = F.gelu(self.c1(f)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        return 2.5 * torch.tanh(self.out(h) / 2.5)


class UncertaintyHead(nn.Module):
    def __init__(self, cin=7, width=24):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(),
                                 nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, f):
        return self.net(f)


INIT_SCHEDULE = ((6, 8.0, 8.0, 1.0, False), (6, 4.0, 4.0, 0.7, False),
                 (6, 4.0, 4.0, 0.7, True), (8, 1.5, 1.5, 0.5, True))


def init_motion(y, schedule=INIT_SCHEDULE):
    """Non-learned coarse-to-fine registration shared by all motion-aware methods: envelope
    (phase-free) Gauss-Newton steps with wide windows, then complex-IQ Gauss-Newton refinement, each
    against the temporal-mean reference in reference coordinates."""
    B, T, H, W = y.shape
    u = torch.zeros(B, T, 2, H, W, device=y.device)
    ones = torch.ones(B, 1, H, W, device=y.device)
    for n_it, win, sm, smt, use_phase in schedule:
        for _ in range(n_it):
            Wp = ops.Warp(u); Pu = ops.phase(u)
            ref = Wp.adj(Pu.conj() * y).mean(1, keepdim=True).expand(B, T, H, W)
            u = ops.motion_update(y, Wp.fwd(ref), u, ones, win, 1.0, use_phase=use_phase)
            u = ops.regularize_motion(u, sm, smt)
    return u.detach()


class UnrolledJMTF(nn.Module):
    def __init__(self, K=6, T=32, flags=None, learn=True, fixed=None):
        super().__init__()
        self.f = dict(FULL); self.f.update(flags or {})
        self.K = K; self.learn = learn
        fx = dict(tau=0.03, lam=0.6, aL=0.95, aS=0.95, eta=0.3, win=2.0, smt=0.5, sm=1.5, beta=0.6)
        fx.update(fixed or {})
        P = lambda v: nn.Parameter(torch.full((K,), float(v)), requires_grad=learn)
        self.log_aL = P(math.log(fx['aL'])); self.log_aS = P(math.log(fx['aS']))
        self.log_tau = P(math.log(fx['tau'])); self.log_lam = P(math.log(fx['lam']))
        th = torch.full((K, T), math.log(1.0)); th[:, :fx.get('keep', 3)] = math.log(1e-3)
        self.sv_th = nn.Parameter(th, requires_grad=learn)
        self.log_eta = P(math.log(fx['eta'])); self.log_win = P(math.log(fx['win']))
        self.log_smt = P(math.log(fx['smt'])); self.log_sm = P(math.log(fx['sm']))
        self.beta_l = P(math.log(fx['beta'] / (1 - fx['beta'])))
        self.psf_a = nn.Parameter(torch.ones(2), requires_grad=learn)
        self.psf_b = nn.Parameter(torch.zeros(2), requires_grad=learn)
        self.pnet = ParamNet() if (self.f['adaptive'] and learn) else None
        self.uhead = UncertaintyHead() if (self.f['uncertainty'] and learn) else None

    # --------------------------------------------------------------- helpers
    def _features(self, E, res, u, Lp, k, sref2):
        lE = torch.log10(E / sref2 + 1e-4)
        v = ops.vesselness(lE)
        lr = torch.log10(res / sref2 + 1e-4)
        mag = u.pow(2).sum(2).mean(1, keepdim=True).sqrt()
        dmag = (u[:, 1:] - u[:, :-1]).pow(2).sum(2).mean(1, keepdim=True).sqrt()
        lL = torch.log10(Lp / Lp.mean((2, 3), keepdim=True) + 1e-4)
        st = torch.full_like(E, k / max(self.K - 1, 1))
        return torch.cat([lE, v, lr, mag, 4 * dmag, lL, st], 1)

    def forward(self, y, psf_true=None, return_all=False, u0=None):
        B, T, H, W = y.shape
        f = self.f
        scale = y.abs().pow(2).mean((1, 2, 3), keepdim=True).sqrt()
        y = y / scale
        # PSF (relative to the target resolution sigma0)
        if f['psf'] == 'none':
            tf = None
            sig = None
        else:
            if f['psf'] == 'oracle':
                sz, sx = psf_true[:, 0], psf_true[:, 1]
            else:
                sz0, sx0 = ops.psf_autocorr(y)
                sz = self.psf_a[0] * sz0 + self.psf_b[0]; sx = self.psf_a[1] * sx0 + self.psf_b[1]
            sig = torch.stack([sz, sx], 1)
            rz = torch.sqrt((sz ** 2 - ops.SIGMA0 ** 2).clamp_min(1e-2))
            rx = torch.sqrt((sx ** 2 - ops.SIGMA0 ** 2).clamp_min(1e-2))
            tf = ops.gauss_tf(H, W, rz, rx, y.device)
        bl = (lambda X: ops.blur(X, tf)) if tf is not None else (lambda X: X)
        # initial state (the coarse registration is non-learned and may be pre-computed)
        if f['motion']:
            u = init_motion(y) if u0 is None else u0
        else:
            u = torch.zeros(B, T, 2, H, W, device=y.device)
        if f['motion']:   # back-project to reference coordinates
            Wp = ops.Warp(u)
            Z0 = Wp.adj(ops.phase(u).conj() * y) if f['phase_model'] else Wp.adj(y)
        else:
            Z0 = y
        L = ops.lowrank_project(Z0, 3)
        sref2 = (Z0 - L).abs().pow(2).mean(1, keepdim=True).flatten(1).median(1).values.view(B, 1, 1, 1).detach()
        sref = sref2.sqrt()
        S = torch.zeros_like(y)
        omega = torch.ones(B, 1, H, W, device=y.device)
        hist, uhist = [], []
        for k in range(self.K):
            ud = u.detach()
            Wp = ops.Warp(ud) if f['motion'] else None
            Pu = ops.phase(ud) if (f['motion'] and f['phase_model']) else None

            def M(Z):          # motion + axial phase (applies to both components)
                if Wp is not None:
                    Z = Wp.fwd(Z)
                if Pu is not None:
                    Z = Pu * Z
                return Z

            def MH(R):
                if Pu is not None:
                    R = Pu.conj() * R
                if Wp is not None:
                    R = Wp.adj(R)
                return R

            # forward model: y = P_u W_u (L + h * S) + n  (tissue kept at the observed resolution,
            # blood reconstructed at the target resolution through the relative PSF h)
            A = lambda Lx, Sx: M(Lx + bl(Sx))
            # data-consistency residual and reliability weights (iteratively re-weighted)
            r = y - A(L, S)
            if f['weights']:
                e = ops.smooth2d(r.abs().pow(2).mean(1, keepdim=True), 4.0)
                omega = (e.flatten(1).median(1).values.view(B, 1, 1, 1) / (e + 1e-12)).clamp(0.25, 4.0)
                omega = omega / omega.amax((2, 3), keepdim=True)
            # tissue: gradient step + learned spectral proximal map
            Lz = L + 0.5 * torch.exp(self.log_aL[k]) * MH(omega * r)
            if self.f.get('nuclear', False):
                Ln = ops.svt_rel(Lz, torch.exp(self.log_tau[k]))
            else:
                Ln = ops.svt_spec(Lz, torch.exp(self.sv_th[k]), smooth=self.learn)
            r = r - M(Ln - L)                                    # Gauss-Seidel residual update
            L = Ln
            # blood: gradient step + adaptive, vessel-preserving structured shrinkage
            Z = S + 0.5 * torch.exp(self.log_aS[k]) * bl(MH(omega * r))
            E = Z.abs().pow(2).mean(1, keepdim=True)
            if f['structured']:
                beta = torch.sigmoid(self.beta_l[k])
                Es = (1 - beta) * E + beta * ops.oriented_energy(E)
            else:
                Es = E
            lam = torch.exp(self.log_lam[k]) * sref
            if self.pnet is not None:
                res = r.abs().pow(2).mean(1, keepdim=True)
                Lp = L.abs().pow(2).mean(1, keepdim=True)
                lam = lam * torch.exp(self.pnet(self._features(E, res, ud, Lp, k, sref2)))
            lam = lam.expand(B, 1, H, W)
            if f['group']:
                nrm = torch.sqrt(Es + 1e-10)
            else:
                nrm = Z.abs() + 1e-10
            if self.learn:   # smoothed shrinkage (never exactly flat, so gradients reach every pixel)
                S = Z * (0.1 * lam * F.softplus((nrm - lam) / (0.1 * lam))) / nrm
            else:            # exact soft (group) thresholding
                S = Z * F.relu(1 - lam / nrm)
            # displacement: Gauss-Newton refinement against the static tissue reference
            if f['motion']:
                G = Wp.fwd(L.mean(1, keepdim=True).expand(B, T, H, W))
                yt = y - M(bl(S))
                du = ops.motion_update(yt.detach(), G.detach(), u.detach(), omega.detach(),
                                       torch.exp(self.log_win[k]), 1.0) - u.detach()
                du = ops.regularize_motion(du, torch.exp(self.log_sm[k]), torch.exp(self.log_smt[k]))
                u = u + torch.exp(self.log_eta[k]) * du        # regularised increment
                uhist.append(u)
            hist.append(S.abs().pow(2).mean(1, keepdim=True))
        pd = hist[-1]
        out = dict(S=S * scale, L=L * scale, u=u, pd=pd * scale.pow(2), sigma=sig, lam=lam, uhist=uhist)
        if self.uhead is not None:
            prev = hist[-2] if len(hist) > 1 else hist[-1]
            res = r.abs().pow(2).mean(1, keepdim=True)
            lp = torch.log10(pd / sref2 + 1e-4); lq = torch.log10(prev / sref2 + 1e-4)
            mag = u.pow(2).sum(2).mean(1, keepdim=True).sqrt()
            feat = torch.cat([lp, lq, (lp - lq).abs(), torch.log10(lam / sref + 1e-4),
                              torch.log10(res / sref2 + 1e-4), mag, ops.vesselness(lp)], 1)
            out['logvar'] = self.uhead(feat.detach())
        if return_all:
            out['hist'] = [h * scale.pow(2) for h in hist]
        return out


# ------------------------------------------------------------------------------- baselines
class CConv(nn.Module):
    """Complex 2-D convolution applied frame-wise (real weights on real/imag channels)."""

    def __init__(self, k=5):
        super().__init__()
        self.c = nn.Conv2d(2, 2, k, padding=k // 2, bias=False)
        with torch.no_grad():
            self.c.weight.zero_(); self.c.weight[0, 0, k // 2, k // 2] = 1; self.c.weight[1, 1, k // 2, k // 2] = 1

    def forward(self, x):
        B, T, H, W = x.shape
        xr = torch.view_as_real(x.reshape(B * T, H, W)).permute(0, 3, 1, 2)
        o = self.c(xr).permute(0, 2, 3, 1).contiguous()
        return torch.view_as_complex(o).view(B, T, H, W)


class DeepUnfoldedRPCA(nn.Module):
    """Convolutional unfolding of RPCA-ISTA in the spirit of CORONA (no motion, no PSF model)."""

    def __init__(self, K=6, T=32):
        super().__init__()
        self.K = K
        self.P1 = nn.ModuleList([CConv() for _ in range(K)]); self.P2 = nn.ModuleList([CConv() for _ in range(K)])
        self.P3 = nn.ModuleList([CConv() for _ in range(K)]); self.P4 = nn.ModuleList([CConv() for _ in range(K)])
        self.log_tau = nn.Parameter(torch.full((K,), math.log(0.03)))
        self.log_lam = nn.Parameter(torch.full((K,), math.log(0.6)))

    def forward(self, y, psf_true=None, return_all=False, u0=None):
        B, T, H, W = y.shape
        scale = y.abs().pow(2).mean((1, 2, 3), keepdim=True).sqrt(); y = y / scale
        L = ops.lowrank_project(y, 3); S = torch.zeros_like(y)
        sref = (y - L).abs().pow(2).mean(1, keepdim=True).flatten(1).median(1).values.view(B, 1, 1, 1).sqrt()
        for k in range(self.K):
            Lt = L + self.P1[k](y - L - S)
            St = S + self.P3[k](y - L - S)
            L = ops.svt_rel(self.P2[k](Lt), torch.exp(self.log_tau[k]))
            Z = self.P4[k](St)
            lam = torch.exp(self.log_lam[k]) * sref
            nrm = Z.abs() + 1e-10
            S = Z * (0.1 * lam * F.softplus((nrm - lam) / (0.1 * lam))) / nrm
        pd = S.abs().pow(2).mean(1, keepdim=True)
        return dict(S=S * scale, L=L * scale, pd=pd * scale.pow(2), u=None)


class DirectCNN(nn.Module):
    """Black-box mapping from the IQ ensemble to the log power-Doppler map (small U-Net)."""

    def __init__(self, T=32, w=32):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())
        self.e1 = blk(T + 1, w); self.e2 = blk(w, 2 * w); self.e3 = blk(2 * w, 4 * w)
        self.d2 = blk(6 * w, 2 * w); self.d1 = blk(3 * w, w); self.o = nn.Conv2d(w, 1, 1)

    def forward(self, y, psf_true=None, return_all=False, u0=None):
        B, T, H, W = y.shape
        scale = y.abs().pow(2).mean((1, 2, 3), keepdim=True).sqrt(); yn = y / scale
        yn = yn - yn.mean(1, keepdim=True)
        # per-pixel log Doppler spectrum (tissue occupies low, blood higher slow-time frequencies)
        spec = torch.fft.fft(yn, dim=1).abs().pow(2)
        x = torch.log10(spec + 1e-8)
        x = torch.cat([x, torch.log10(yn.abs().pow(2).mean(1, keepdim=True) + 1e-8)], 1)
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)
        a = self.e1(x); b = self.e2(F.avg_pool2d(a, 2)); c = self.e3(F.avg_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        e = self.d1(torch.cat([F.interpolate(d, scale_factor=2), a], 1))
        logpd = self.o(e)
        logpd = (logpd - logpd.mean((2, 3), keepdim=True)).clamp(-8, 8)   # relative log map (gain is not identifiable)
        sref2 = yn.abs().pow(2).mean(1, keepdim=True).flatten(1).median(1).values.view(B, 1, 1, 1)
        pd = 10 ** logpd * sref2 * scale.pow(2)
        return dict(pd=pd, u=None, S=None)
