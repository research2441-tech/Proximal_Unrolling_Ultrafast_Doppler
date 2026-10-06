# Physics-informed proximal unrolling for ultrafast Doppler (joint tissue, motion and microvascular flow)

Reference implementation of the manuscript "Physics-Informed Proximal Unrolling with Spatially Adaptive Priors for Joint Tissue, Motion and Microvascular Flow Reconstruction in Ultrafast Doppler Ultrasound". All files are in one folder. Scripts create the working sub-folders `data/`, `runs/`, `results/` and `figs/` when they run.

## Files
| File | Purpose |
|---|---|
| `sim.py` | In silico ultrafast Doppler simulator: tissue speckle, vessel network, AR(1) blood, rigid, non-rigid and strain motion, axial phase, Gaussian PSF, heteroscedastic noise |
| `ops.py` | Differentiable operators: cubic B-spline warp with exact adjoint, phase, FFT PSF, PSF autocorrelation estimator, complex Gauss–Newton motion step, spectral shrinkage, oriented energy, vesselness |
| `model.py` | Unrolled network (`UnrolledJMTF`), coarse-to-fine registration (`init_motion`), learned baselines (DU-RPCA, Direct CNN) |
| `baselines.py` | SVD, Doppler-adaptive SVD, RPCA (inexact ALM), motion-compensated SVD, joint proximal scheme (JPA) |
| `data.py` | Generates training, validation, test and 18 stress cohorts, with cached coarse registration |
| `recompute_u0.py` | Recomputes the cached registration for all stored cohorts |
| `train.py` | Training: `python train.py <variant> <seed> [iters]` (variants: full, no_adapt, no_struct, l1_sparse, no_psf, oracle_psf, no_motion, no_weights, nuclear, K3, K9, durpca, cnn) |
| `evaluate.py` | `python evaluate.py tune` (baseline grids on validation), `python evaluate.py run <cohort ...>` |
| `tune_extra.py`, `tune_jpa2.py` | Grid extensions used for interior optima of A-SVD and JPA |
| `eval_nw.py` | Evaluates the variant without reliability weights on hard cohorts |
| `metrics.py` | NRMSE, PSNR, SSIM, CR, vSNR, AUC, microvessel/slow-flow recall, EPE, calibration, sparsification |
| `analysis.py`, `extra_analysis.py`, `lam_stats.py`, `timing.py` | Statistics (Wilcoxon, Holm, bootstrap), uncertainty/conformal analysis, stage analysis, PSF accuracy, weight-map statistics, timing |
| `fig_arch.py`, `fig_shift.py` | Figure scripts (architecture, generalisation/efficiency). The result figures come from `analysis.py` and `extra_analysis.py` |
| `*.pt` | Trained checkpoints (`<variant>_s<seed>.pt`) |
| `*.json` | Result files behind every table and figure (per-sequence metrics for each method and cohort, tuning, uncertainty, stages, PSF, timing, weight-map statistics) |
| `fig*.png` | Manuscript figures |

## Reproduction
```bash
pip install -r requirements.txt
python data.py val test train stress      # simulate cohorts (about 20 min on 2 cores)
python recompute_u0.py                    # cache the coarse registration
python evaluate.py tune && python tune_extra.py && python tune_jpa2.py
for s in 0 1 2 3 4; do python train.py full $s; done   # plus the other variants
python evaluate.py run val test motion0 motion05 motion10 motion15 motion20 bnr5 bnr10 bnr15 bnr20 ctr20 ctr30 ctr40 ctr45 psfwide tubes penetrating decor slowflow
python eval_nw.py bnr5 ctr45 motion20 psfwide penetrating decor
python analysis.py; python extra_analysis.py unc stages psf; python lam_stats.py; python timing.py
python fig_arch.py; python fig_shift.py
```
To reuse the provided checkpoints and results without retraining, run `mkdir -p runs results figs && mv *.pt runs/ && mv *.json results/` first.

Hardware used: two CPU threads, no GPU. PyTorch 2.14. Seeds are fixed in every script.
