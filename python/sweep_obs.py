"""Output feedback, final pass.
omega = 0: all sensor-noise levels, refined tuning of the stationary LQG.
omega > 0: the flow is in statistical equilibrium under the forcing (and under the stationary
compensator, or the standard LQG for the time-varying ones) when the burst appears at t = 0."""
import common
import numpy as np, time, pickle, sys
from scipy.optimize import minimize_scalar
from lq import Plant, LQProblem, lyap
from obs import wall_shear_C, Horizon, StationaryLQG, output_power, kalman_stationary

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
NUS0 = np.array([1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
NUSW = np.array([1e-4, 1e-2, 1.0])
OMSW = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
OMD = np.r_[0.0, np.geomspace(1e-2, 1e4, 7)]

def tune(S, C, W, V, Sigma0, Wshape):
    Jb, ob, Js = S.best(C, W, V, Sigma0, Wshape, OMD)
    if ob > 0:
        f = lambda lg: S.cost(C, W, V, Sigma0, kalman_stationary(S.lq.A, C, W + 10**lg * Wshape, V)[1])
        r = minimize_scalar(f, bounds=(np.log10(ob) - 1, np.log10(ob) + 1), method='bounded', options={'xatol': 0.05})
        if r.fun < Jb:
            Jb, ob = r.fun, 10**r.x
    return Jb, ob, Js

def run(case, act, TF, rhat=1.0, nq=120, check=False):
    lx, lz = CASES[case]
    t0 = time.time()
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    C = wall_shear_C(pl); m = C.shape[0]
    Py = output_power(pl, C, z0, tmax)
    Sigma0 = np.outer(z0, z0.conj())
    Zp = pl.Zphys; nS = Zp.shape[1]
    Wiso = Zp @ Zp.conj().T / (nS * tmax); Wbur = Sigma0 / tmax
    T = TF * tmax
    S = StationaryLQG(lq, T)
    H = Horizon(lq, T, nq=nq)
    n = lq.n
    print(f'{case} {act} {TF}: setup {time.time()-t0:.0f}s', flush=True)
    res = []
    grid = [(0.0, nu) for nu in NUS0] + [(om, nu) for om in OMSW for nu in NUSW]
    for om, nu in grid:
        W = om * Wiso; V = nu * Py * tmax * np.eye(m)
        if om > 0:
            Zunc = lyap(lq.A, W); Zfi = lyap(lq.Ac, W)
            Sinf, Linf, _ = kalman_stationary(lq.A, C, W, V)
            Xss = S.stationary_covariance(W, Linf @ V @ Linf.conj().T, lq.A - Linf @ C)
            Zlqg = Xss[:n, :n]
        else:
            Zunc = Zfi = Zlqg = None
        c = H.output_feedback(C, W, V, Sigma0, Zunc, Zfi, Zlqg)
        if check and om > 0:
            c0 = H.output_feedback(C, W, V, 0 * Sigma0, Zunc, Zfi, Zlqg)
            print(f'   check (no burst): KF+LQR {c0["KF_LQR"]:.6e}  stationary LQG {S.cost(C, W, V, 0*Sigma0, Linf):.6e}', flush=True)
        Jb, ob, Jbs = tune(S, C, W, V, Sigma0, Wbur)
        Ji, oi, Jis = S.best(C, W, V, Sigma0, Wiso, OMD)
        r = dict(omega=om, nu=nu, LQG_burst=Jb, omd_burst=ob, LQG_iso=Ji, omd_iso=oi, LQG_std=Jbs[0],
                 Jbs=Jbs, Jis=Jis, **{k: c[k] for k in ['unc', 'FI_LQR', 'FI_opt', 'KF_LQR', 'opt']})
        if om == 0:
            r['s'] = c['s']; r['trSigma'] = c['trSigma']
        res.append(r)
        print(f"  omega={om:g} nu={nu:g}: unc={c['unc']:.5g} FI_LQR={c['FI_LQR']:.5g} FI_opt={c['FI_opt']:.5g} KF={c['KF_LQR']:.5g} "
              f"opt={c['opt']:.5g} LQGb={Jb:.5g}({ob:.3g}) LQGstd={Jbs[0]:.5g} LQGi={Ji:.5g}({oi:g}) [{time.time()-t0:.0f}s]", flush=True)
    out = dict(case=case, act=act, TF=TF, tmax=tmax, Gmax=Gmax, rho=rho, Py=Py, res=res, NUS=NUS0, NUSW=NUSW, OMS=np.r_[0, OMSW], OMD=OMD)
    pickle.dump(out, open(f'{common.DATA}/obs3_{case}_{act}_T{TF:g}.pkl', 'wb'))
    print('done', case, act, TF, time.time()-t0, flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], float(sys.argv[3]), check=(len(sys.argv) > 4))
