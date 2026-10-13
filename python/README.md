# Linear vs unrestricted optimal control of the eddy-viscosity Orr–Sommerfeld–Squire model

Python port (and extension) of the MATLAB codes in `../matlab` used for the paper
"How optimal is linear control?". The results of all the runs are in `data/`.

## Files

| file | content |
|---|---|
| `orrss.py` | Port of `Dmat.m`, `turprof_generic.m`, `turchan.m`, `prenorm.m`, `optimalu2.m`; accurate (spectral) mean profile / eddy viscosity; `OrrModel` class; wall-transpiration actuator (through the relaxation boundary rows, as in `turchan.m`). |
| `lq.py` | Energy-coordinate plant (`Plant`), fast Riccati solver (Hamiltonian Schur + Newton–Kleinman), LQR, closed-form finite-horizon optimum (Proposition 1 of the paper), direct L-BFGS optimisation of the control history, discrete Riccati recursion, bounded actuation (FISTA) and saturated LQR, discounted (random-lifetime) costs. |
| `common.py` | Global parameters (Re_tau = 2000, N = 128, full eddy viscosity) and data directory. |
| `validate_port.py` | Reproduces `../matlab/New_res2e3_0.mat` (MATLAB) to 8 digits: uncontrolled and LQR energy histories. |
| `sweep_tg.py` | Transient-growth map. |
| `sweep_maps.py` | Maps of the LQR optimality ratio over (lambda_x, lambda_z) (`data/maps_20x20.pkl`). |
| `cases.py` | Representative cases: histories, fields, scans in T and rho, verification, bounded actuation (`data/case_*.pkl`). |
| `sensitivity.py` | Resolution / actuator time constant / eddy-viscosity sensitivity (`data/sensitivity.pkl`). |
| `obs.py` | Output feedback: wall shear-stress sensors, Kalman-Bucy filters (stationary and time-varying, closed form), exact expected costs of the optimal output feedback (separation theorem), of KF + LQR and of stationary LQG compensators (block Lyapunov/Sylvester), statistically stationary background. |
| `sweep_obs.py` | Sensor-noise / process-noise study for the representative cases (`data/obs3_<case>_<actuator>_T<T/tmax>.pkl`). |
| `intermittent.py` | Wall-pressure sensors; Bernoulli-Gaussian (intermittent) bursts: matched-filter statistic, Kalman and nonlinear conditional-mean estimators, exact mean-square errors (1-D quadrature, checked by Monte Carlo). |
| `sweep_intermittent.py` | Intermittent-burst study (`data/int_<case>_<actuator>_T<T/tmax>.pkl`). |
| `drag.py` | FIK drag matrix, drag objective with amplitude-proportional (l1) actuation cost, accelerated proximal gradient, value function, expected costs of the nonlinear optimum and of the best linear feedback. |
| `sweep_drag.py` | Drag study (`data/drag_<case>_<actuator>_T<T/tmax>.pkl`). |
| `pdrag.py` | Drag objective with amplitude-proportional cost from noisy wall-pressure measurements: exact input-space formulation, best linear output feedback (exact expected cost, convex), Monte Carlo evaluation of thresholded certainty-equivalent and certainty-equivalent MPC controllers (Kalman and conditional-mean estimates). |
| `sweep_pdrag.py` | Partial-information drag study (`data/pdrag_<case>_p<p>_nu<nu>.pkl`). |
| `pdrag_extra.py` | Direct policy search (sample-average approximation, JAX + L-BFGS) of a dead-zone output feedback for the same problem (`data/pdragps_*.pkl`). |
| `pdrag_summary.py` | Text summary of the partial-information drag results. |
| `misestimation.py` | Controllers designed for the wrong (deterministic or random) lifetime (`data/mis_<case>_<actuator>.pkl`). |
| `reynolds.py`, `run_re*.sh`, `re_table.py`, `fill_retable.py` | Reynolds-number dependence, Re_tau = 550-10^4, with resolution checks (`data/re_*.pkl`); rows of the appendix table, inserted into the LaTeX source by `fill_retable.py <paper.tex>`. |
| `nconv.py` | Fine resolution scan N = 96-256 (`data/nconv_<case>.pkl`). |
| `qlmodel.py` | Quasi-linear model: one Fourier mode interacting with the mean-flow distortion, ETD-RK2 time stepper in JAX (automatic differentiation). |
| `ql_study.py` | Linear-model controllers (LQR, finite-horizon feedback, re-tuned penalty) vs nonlinear optimum (L-BFGS on the control history) in the quasi-linear model (`data/ql_*.pkl`, `data/qltuned_*.pkl`). |
| `ql_diag.py`, `ql_conv.py`, `ql_check192.py`, `ql_summary.py` | Amplitude in physical units and mean-flow distortion; time-step/resolution convergence (uncontrolled and LQR; full comparison at N = 192); text summary. |
| `pulses.py` | Best trains of K wall-transpiration pulses (optimal amplitudes and timing) in the linear model, compared with the LQR and the optimum (`data/pulses_*.pkl`). |
| `ql_pulses.py` | Restarts of the quasi-linear nonlinear optimisation from pulse-like controls (`data/qlpulses_*.pkl`). |
| `run_queue.sh`, `pin_jax.sh` | Minimal job queue with bounded concurrency; pins the multithreaded JAX jobs to a subset of cores. |

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1        # essential: multithreaded BLAS is ~100x slower here (set it before numpy is imported)
python validate_port.py
python sweep_maps.py 20 20           # ~1.5 h on 18 cores
python cases.py outer_wall inner_wall outer_distributed inner_distributed
python sensitivity.py
for c in inner outer; do for a in wall distributed; do for T in 1 2; do python sweep_obs.py $c $a $T & done; done; done   # ~1 h
for c in inner outer; do for a in wall distributed; do for T in 1 2; do python sweep_intermittent.py $c $a $T & python sweep_drag.py $c $a $T & done; done; done   # ~25 min
for c in inner outer; do for a in wall distributed; do python misestimation.py $c $a & done; done          # ~10 min
for c in inner outer; do for p in 1 0.1; do for nu in 0.01 0.1 1; do python sweep_pdrag.py $c $p $nu & done; done; done   # ~2.5 h
for c in inner outer; do for p in 1 0.1; do for nu in 0.01 0.1 1; do python pdrag_extra.py $c $p $nu; done; done; done    # JAX
./run_re.sh; ./run_re2.sh; ./run_re3.sh; python nconv.py inner; python nconv.py outer
python ql_diag.py inner; python ql_diag.py outer
python ql_study.py inner wall 1 full 0.05,0.1,0.15,0.2,0.3,0.5,1.0,1.5,2.0      # likewise for T=2, the other actuator and case
python ql_study.py inner wall 1 tuned 0.05,0.1,0.15,0.2,0.3,0.5,1.0,1.5,2.0
python ../figures/make_all.py       # figures (one script per figure in ../figures)
python re_table.py
```

## Note on the earlier MATLAB "non-linear optimum"

`New_res2e3_0.mat` stores an fminunc result with `fval = 0.315` for a control history that is
numerically zero (|sol| ~ 1e-15), whereas the exact uncontrolled cost of that case is 0.891
(`validate_port.py`). The saved optimum is therefore an artefact of the cost evaluation in
`aMinimizar.m`/`ode23`, and it should not be used. The present direct optimisation uses exact
zero-order-hold propagation, a discrete adjoint gradient and L-BFGS, and is verified against the
closed-form optimum.
