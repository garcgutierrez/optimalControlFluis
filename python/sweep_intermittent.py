"""Intermittent bursts, wall-pressure sensors: linear (Kalman) versus nonlinear (conditional-mean)
estimation with the optimal finite-horizon gain, and the best stationary LQG."""
import common
import numpy as np, time, pickle, sys
import scipy.linalg as sla
from scipy.integrate import cumulative_trapezoid
from lq import Plant, LQProblem
from obs import Horizon, StationaryLQG, output_power
from intermittent import wall_pressure_C, mse_linear, mse_nonlinear
from sweep_obs import tune

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
PS = np.array([1.0, 0.3, 0.1, 0.03, 0.01])
NUS = np.geomspace(1e-4, 1e2, 13)
PS_LQG = [1.0, 0.1, 0.01]
NUS_LQG = [1e-3, 1e-2, 1e-1, 1.0]

def run(case, act, TF, rhat=1.0, nq=120):
    lx, lz = CASES[case]
    t0 = time.time()
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    C = wall_pressure_C(pl); m = C.shape[0]
    Pp = output_power(pl, C, z0, tmax)
    T = TF * tmax
    H = Horizon(lq, T, nq=nq)
    # weights w(s) = z_unc^H Po S Po z_unc and full-information costs per unit p
    zs = [E @ z0 for E in H.expA]
    w = np.array([float(np.real(z.conj() @ X @ z)) for z, X in zip(zs, H.XO)])
    J0 = float(np.real(z0.conj() @ H.W0[0] @ z0)); JFI = float(np.real(z0.conj() @ H.Po[0] @ z0))
    JFIL = float(np.real(z0.conj() @ H.PL[0] @ z0))
    # accumulated signal for nu = 1:  h1(s) = int |C z_unc|^2 dr / (Pp tmax)
    td = np.linspace(0, T, 8001); E1 = sla.expm(pl.A * (td[1] - td[0]))
    z = z0.copy(); sig = np.zeros(len(td))
    for j in range(len(td)):
        sig[j] = float(np.sum(np.abs(C @ z) ** 2)); z = E1 @ z
    h1 = np.interp(H.s, td, cumulative_trapezoid(sig, td, initial=0.0)) / (Pp * tmax)
    print(f'{case} {act} {TF}: setup {time.time()-t0:.0f}s  h1(T)={h1[-1]:.3f}', flush=True)
    res = []
    for p in PS:
        for nu in NUS:
            h = h1 / nu
            mL = np.array([mse_linear(p, hk) for hk in h]); mN = np.array([mse_nonlinear(p, hk) for hk in h])
            r = dict(p=p, nu=nu, unc=p*J0, FI_opt=p*JFI, FI_LQR=p*JFIL,
                     KF=p*JFI + H.w @ (w * mL), NL=p*JFI + H.w @ (w * mN))
            res.append(r)
        print(f'  p={p:g} done [{time.time()-t0:.0f}s]', flush=True)
    S = StationaryLQG(lq, T)
    Sig0 = np.outer(z0, z0.conj())
    lqg = []
    for p in PS_LQG:
        for nu in NUS_LQG:
            V = nu * Pp * tmax * np.eye(m)
            Jb, ob, _ = tune(S, C, 0 * Sig0, V, p * Sig0, Sig0 / tmax)
            lqg.append(dict(p=p, nu=nu, LQG=Jb, omd=ob))
            print(f'  LQG p={p:g} nu={nu:g}: {Jb:.5g} ({ob:.3g}) [{time.time()-t0:.0f}s]', flush=True)
    out = dict(case=case, act=act, TF=TF, tmax=tmax, Gmax=Gmax, rho=rho, Pp=Pp, s=H.s, w=w, h1=h1,
               res=res, lqg=lqg, PS=PS, NUS=NUS)
    pickle.dump(out, open(f'{common.DATA}/int_{case}_{act}_T{TF:g}.pkl', 'wb'))
    print('done', case, act, TF, time.time()-t0, flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], float(sys.argv[3]))
