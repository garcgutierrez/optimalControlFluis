"""Quasi-linear model: are there pulse-like controls better than the nonlinear optimum found in ql_study?
The problem is non-convex, so the L-BFGS optimisation of the control history is restarted from pulsed,
bang-bang-like and random-pulse initial controls, and the costs are compared with the reported optimum."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
import common
import numpy as np, pickle, sys, time, glob
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from lq import Plant, Discretised, discrete_riccati_optimum
from qlmodel import QLModel

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}


def pulse_train(U, Kp, width, rng=None, scale=1.0):
    """Kp pulses of `width` intervals that carry the same impulse as U over the lifetime."""
    Nt = U.shape[0]
    if rng is None:
        centres = ((np.arange(Kp) + 0.5) * Nt / Kp).astype(int)
    else:
        centres = np.sort(rng.choice(Nt - width, Kp, replace=False))
    P = np.zeros_like(U)
    edges = np.r_[0, (centres[1:] + centres[:-1]) // 2, Nt]
    for k, c in enumerate(centres):
        s = max(0, min(c, Nt - width))
        imp = U[edges[k]:edges[k + 1]].sum(0)               # impulse of U in the pulse's share of the lifetime
        P[s:s + width] = scale * imp / width
    return P


def run(case, TF, eps_list, Nt=300, N=128, rhat=1.0):
    t0 = time.time()
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator='wall', profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * tmax; T = TF * tmax
    ql = QLModel(pl); ql.build(T / Nt)
    disc = Discretised(pl.A, pl.B, T, Nt)
    Ulin, _, _ = discrete_riccati_optimum(disc, z0, rho)
    m = pl.m; shape = (Nt, m)
    sim = jax.jit(lambda z, U: ql.simulate(z, U, rho)[0])
    rec = []
    for f in glob.glob(f'{common.DATA}/ql_{case}_wall_T{TF:g}*.pkl'):
        rec += pickle.load(open(f, 'rb'))['res']
    out = []
    for eps in eps_list:
        z = eps * z0
        Jrep = [r['NLopt'] for r in rec if np.isclose(r['eps'], eps)][0]
        f = lambda x: sim(z, (x[:Nt*m] + 1j * x[Nt*m:]).reshape(shape))
        vg = jax.jit(jax.value_and_grad(f))
        def fg(x):
            v, g = vg(jnp.asarray(x)); return float(v), np.asarray(g, dtype=float)
        U = eps * Ulin
        rng = np.random.default_rng(1)
        one = np.zeros_like(U); one[:6] = U.sum(0) / 6                 # whole impulse in the first 2% of the life
        inits = {'one pulse (first 2%)': one,
                 '3 pulses': pulse_train(U, 3, 6), '10 pulses': pulse_train(U, 10, 3),
                 'bang-bang modulus': np.abs(U).max() * np.exp(1j * np.angle(U)),
                 'random pulses a': pulse_train(U, 5, 4, rng, 2.0), 'random pulses b': pulse_train(U, 5, 4, rng, 0.5)}
        for name, U0 in inits.items():
            shrink = 1.0
            Jinit = float(sim(z, U0))
            while not np.isfinite(Jinit) and shrink > 1e-3:   # pulses too strong for the time stepper: weaken them
                U0 = 0.5 * U0; shrink *= 0.5; Jinit = float(sim(z, U0))
            x0 = np.concatenate([U0.real.ravel(), U0.imag.ravel()])
            r = minimize(fg, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=3000, maxcor=30, ftol=1e-14, gtol=1e-11))
            out.append(dict(eps=eps, init=name, Jinit=Jinit, Jfinal=r.fun, Jrep=Jrep, nit=r.nit, shrink=shrink))
            print(f'{case} wall T={TF:g} eps={eps}: init "{name}" (amplitude x{shrink:g}): pulse control itself J/J_NL={Jinit/Jrep:.4f}; '
                  f'after optimisation J/J_NL={r.fun/Jrep:.6f} (its={r.nit}) [{time.time()-t0:.0f}s]', flush=True)
    pickle.dump(out, open(f'{common.DATA}/qlpulses_{case}_T{TF:g}_e{eps_list[0]:g}.pkl', 'wb'))
    print('done', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]), [float(e) for e in sys.argv[3].split(',')])
