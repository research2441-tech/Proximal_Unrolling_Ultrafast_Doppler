"""Non-learned baselines: SVD, Doppler-adaptive SVD, convex RPCA (inexact ALM), motion-compensated SVD
and a joint proximal alternating scheme (re-implementation of the convex joint formulation)."""
import numpy as np
import torch
import ops
import model


def _norm(y):
    return y / y.abs().pow(2).mean((1, 2, 3), keepdim=True).sqrt()


def svd_filter(y, k):
    """y: (B,T,H,W) complex tensor -> PD (B,1,H,W)."""
    y = _norm(y)
    F = y - ops.lowrank_project(y, k)
    return F.abs().pow(2).mean(1, keepdim=True)


def asvd_filter(y, fc):
    """Cut-off chosen per sequence as the first temporal singular vector whose power-weighted mean
    Doppler frequency exceeds fc (cycles/frame)."""
    y = _norm(y)
    B, T, H, W = y.shape
    out = []
    for b in range(B):
        C = y[b].reshape(T, -1)
        G = C @ C.conj().T
        ev, V = torch.linalg.eigh(G.to(torch.complex128))
        V = V.flip(-1).to(torch.complex64)               # columns: temporal singular vectors (desc.)
        spec = torch.fft.fft(V, dim=0).abs().pow(2)
        f = torch.fft.fftfreq(T).abs().view(-1, 1)
        mf = (spec * f).sum(0) / spec.sum(0)
        idx = torch.nonzero(mf > fc)
        k = int(idx[0]) if len(idx) else T - 1
        k = max(k, 1)
        P = V[:, :k] @ V[:, :k].conj().T
        Fm = C - P @ C
        out.append(Fm.abs().pow(2).mean(0).reshape(1, H, W))
    return torch.stack(out)


def rpca(y, c, iters=60):
    """Convex RPCA min ||L||_* + lam ||S||_1 s.t. Y = L + S (inexact augmented Lagrangian)."""
    y = _norm(y)
    B, T, H, W = y.shape
    N = H * W
    lam = c / np.sqrt(max(N, T))
    Y = y
    L = torch.zeros_like(Y); S = torch.zeros_like(Y); Z = torch.zeros_like(Y)
    nrm2 = torch.linalg.matrix_norm(Y.reshape(B, T, N), ord=2).view(B, 1, 1, 1)
    mu = 1.25 / nrm2; rho = 1.5
    for _ in range(iters):
        X = Y - S + Z / mu
        Cm = X.reshape(B, T, N)
        G = Cm @ Cm.conj().transpose(1, 2)
        ev, V = torch.linalg.eigh(G.to(torch.complex128))
        s = ev.real.clamp_min(0).sqrt().to(torch.float32)
        V = V.to(Cm.dtype)
        shrink = torch.relu(1 - (1 / mu.view(B, 1)) / (s + 1e-12))
        L = (V @ torch.diag_embed(shrink.to(Cm.dtype)) @ V.conj().transpose(1, 2) @ Cm).reshape(B, T, H, W)
        X = Y - L + Z / mu
        S = X * torch.relu(1 - (lam / mu) / (X.abs() + 1e-12))
        Z = Z + mu * (Y - L - S)
        mu = mu * rho
    return S.abs().pow(2).mean(1, keepdim=True)


def mc_svd(y, u0, k):
    y = _norm(y)
    W = ops.Warp(u0)
    Z = W.adj(ops.phase(u0).conj() * y)
    F = Z - ops.lowrank_project(Z, k)
    return F.abs().pow(2).mean(1, keepdim=True)


def jpa(y, u0, tau, lam, K=30):
    """Joint proximal alternating scheme with nuclear-norm tissue prior, l1 blood prior and smooth
    displacement prior; all weights global and fixed (tuned on validation data)."""
    flags = dict(adaptive=False, structured=False, group=False, psf='none', weights=False,
                 uncertainty=False, nuclear=True)
    m = model.UnrolledJMTF(K=K, flags=flags, learn=False, fixed=dict(tau=tau, lam=lam))
    with torch.no_grad():
        o = m(y, u0=u0)
    return o['pd'], o['u']
