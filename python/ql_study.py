"""Nonlinear (quasi-linear, mean-flow distortion) dynamics: linear-model controllers versus the
nonlinear optimal control computed by direct optimisation of the control history (L-BFGS with
exact gradients from automatic differentiation), as a function of the burst amplitude."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import common
import numpy as np, pickle, sys, time
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from lq import Plant, LQProblem, Discretised, discrete_riccati_optimum, care
from pdrag import lq_gains
from qlmodel import QLModel

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
EPS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

def run(case, act, TF, eps_list=EPS, tag='', Nt=300, N=128, rhat=1.0):
    t0 = time.time()
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator=act, profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    T = TF * tmax
    ql = QLModel(pl); ql.build(T / Nt)
    disc = Discretised(pl.A, pl.B, T, Nt)
    lq = LQProblem(pl.A, pl.B, rho)
    m, n = pl.m, pl.n
    Klqr = jnp.asarray(lq.K)
    Kltv = jnp.asarray(np.array(lq_gains(disc, np.eye(n), rho)))       # finite-horizon (linear model)
    Ulin_unit, _, _ = discrete_riccati_optimum(disc, z0, rho)          # linear optimal open-loop history
    sim = jax.jit(lambda z, U: ql.simulate(z, U, rho))
    simK = jax.jit(lambda z, K: ql.simulate(z, None, rho, feedback=(K, Nt)))   # gains as arguments
    fb_lqr, fb_ltv = (lambda z: simK(z, Klqr)), (lambda z: simK(z, Kltv))
    shape = (Nt, m)
    def fg_factory(z):
        f = lambda x: sim(z, (x[:Nt*m] + 1j * x[Nt*m:]).reshape(shape))[0]
        vg = jax.jit(jax.value_and_grad(f))
        def fg(x):
            v, g = vg(jnp.asarray(x))
            return float(v), np.asarray(g, dtype=float)
        return fg
    res = []
    for eps in eps_list:
        z = eps * z0
        J0 = float(sim(z, np.zeros(shape, complex))[0])
        Jl = fb_lqr(z); Jt = fb_ltv(z)
        Jopen = float(sim(z, eps * Ulin_unit)[0])
        fg = fg_factory(z)
        best = None
        for init in ['linopt', 'zero']:
            U0 = eps * Ulin_unit if init == 'linopt' else np.zeros(shape, complex)
            x0 = np.concatenate([U0.real.ravel(), U0.imag.ravel()])
            r = minimize(fg, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=3000, maxcor=30, ftol=1e-14, gtol=1e-11))
            if best is None or r.fun < best.fun:
                best = r
        Uopt = (best.x[:Nt*m] + 1j * best.x[Nt*m:]).reshape(shape)
        Jn, Ef, Em = sim(z, Uopt)
        _, EfL, EmL = Jl[0], Jl[1], Jl[2]
        rec = dict(eps=eps, J0=J0, LQR=float(Jl[0]), LTV=float(Jt[0]), open=Jopen, NLopt=float(Jn), its=best.nit,
                   EfN=np.asarray(Ef), EmN=np.asarray(Em), EfL=np.asarray(Jl[1]), EmL=np.asarray(Jl[2]), Uopt=Uopt)
        res.append(rec)
        e2 = eps ** 2
        print(f'{case} {act} T={TF} eps={eps}: J0={J0/e2:.5f} LQR={rec["LQR"]/e2:.5f} LTV={rec["LTV"]/e2:.5f} openlin={Jopen/e2:.5f} '
              f'NLopt={rec["NLopt"]/e2:.5f}  LQR/NL={rec["LQR"]/rec["NLopt"]:.4f} LTV/NL={rec["LTV"]/rec["NLopt"]:.4f} its={best.nit} [{time.time()-t0:.0f}s]', flush=True)
    pickle.dump(dict(case=case, act=act, TF=TF, tmax=tmax, rho=rho, Nt=Nt, res=res, t=np.arange(Nt + 1) * T / Nt),
                open(f'{common.DATA}/ql_{case}_{act}_T{TF:g}{tag}.pkl', 'wb'))
    print('done', case, act, TF, time.time() - t0, flush=True)

FAC = 2.0 ** np.arange(-2, 13)          # design penalty rho_d = FAC * rho

def run_tuned(case, act, TF, eps_list=EPS, tag='', Nt=300, N=128, rhat=1.0):
    """LQR and finite-horizon linear feedback designed for the linear model with a re-tuned penalty
    rho_d, applied to the nonlinear model (cost always evaluated with the true rho)."""
    t0 = time.time()
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator=act, profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    T = TF * tmax
    ql = QLModel(pl); ql.build(T / Nt)
    disc = Discretised(pl.A, pl.B, T, Nt)
    simK = jax.jit(lambda z, K: ql.simulate(z, None, rho, feedback=(K, Nt))[0])
    EPSL = np.array(eps_list)
    JL = np.zeros((len(FAC), len(EPSL))); JT = np.zeros((len(FAC), len(EPSL)))
    for i, f in enumerate(FAC):
        P = care(pl.A, pl.B, f * rho); K = jnp.asarray(pl.B.conj().T @ P / (f * rho))
        Kt = jnp.asarray(np.array(lq_gains(disc, np.eye(pl.n), f * rho)))
        for j, eps in enumerate(EPSL):
            JL[i, j] = float(simK(eps * z0, K)); JT[i, j] = float(simK(eps * z0, Kt))
        del Kt
        print(f'  {case} {act} T={TF} f={f:g}: LQR/eps2 {np.round(JL[i] / EPSL**2, 5)}  LTV/eps2 {np.round(JT[i] / EPSL**2, 5)} [{time.time()-t0:.0f}s]', flush=True)
    pickle.dump(dict(case=case, act=act, TF=TF, FAC=FAC, EPS=EPSL, JL=JL, JT=JT),
                open(f'{common.DATA}/qltuned_{case}_{act}_T{TF:g}{tag}.pkl', 'wb'))
    print('done tuned', case, act, TF, time.time() - t0, flush=True)

if __name__ == '__main__':
    # ql_study.py case act TF [full|tuned] [eps1,eps2,...] [tag]
    mode = sys.argv[4] if len(sys.argv) > 4 else 'full'
    eps_list = [float(e) for e in sys.argv[5].split(',')] if len(sys.argv) > 5 else EPS
    tag = sys.argv[6] if len(sys.argv) > 6 else ''
    (run_tuned if mode == 'tuned' else run)(sys.argv[1], sys.argv[2], float(sys.argv[3]), eps_list, tag)
