"""Drag objective with amplitude-proportional actuation cost from noisy wall-pressure measurements:
full information (bounds), best linear output feedback (exact), nonlinear output feedback (Monte Carlo)."""
import common
import numpy as np, time, pickle, sys
import scipy.linalg as sla
from scipy.integrate import cumulative_trapezoid, quad
from lq import Plant, Discretised
from drag import drag_matrix
from intermittent import wall_pressure_C
from obs import output_power
from pdrag import InputSpace, LinearPolicy, MonteCarlo, lq_gains

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
GRS = [0.1, 0.3, 0.7]
TAUF = [0.25, 0.5, 0.75, 1.0, 1.5]

def setup(case, TF=2.0, Nt=60, rhat=1.0):
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator='wall', profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * tmax; T = TF * tmax
    disc = Discretised(pl.A, pl.B, T, Nt); M = drag_matrix(pl)
    I = InputSpace(disc, M, rho, z0)
    C = wall_pressure_C(pl); Pp = output_power(pl, C, z0, tmax)
    td = np.linspace(0, T, 8001); E1 = sla.expm(pl.A * (td[1] - td[0])); z = z0.copy(); sig = np.zeros(len(td))
    for j in range(len(td)):
        sig[j] = np.sum(np.abs(C @ z) ** 2); z = E1 @ z
    h1 = np.interp(np.arange(Nt + 1) * disc.h, td, cumulative_trapezoid(sig, td, initial=0) / (Pp * tmax))
    return pl, tmax, z0, rho, disc, M, I, h1

def full_information(I, gc):
    """Value function on a grid (continuation) and expected FI costs for a ~ CN(0,1)."""
    gs = np.r_[gc * np.geomspace(1.0, 1e-3, 40), 0.0]
    V = []; U = None
    for g in gs:
        v, U = I.value(g, U0=U); V.append(v)
    V = np.array(V)
    lg = np.log(gs[:-1])
    def Vf(g):
        if g >= gc: return I.D0
        if g <= gs[-2]: return V[-1] + (V[-2] - V[-1]) * g / gs[-2]
        return float(np.interp(np.log(g), lg[::-1], V[:-1][::-1]))
    out = {}
    for gr in GRS:
        g = gr * gc
        nl = quad(lambda r: r**2 * Vf(g / r) * 2 * r * np.exp(-r**2) if r > 0 else 0.0, 0, 8, limit=400)[0]
        out[gr] = dict(FI_NL=nl, FI_lin=Vf(g * np.sqrt(np.pi) / 2))
    return out, gs, V

def run(case, p, nu, S=4000):
    t0 = time.time()
    pl, tmax, z0, rho, disc, M, I, h1 = setup(case)
    gc = I.threshold()
    FI, gs, V = full_information(I, gc)
    print(f'{case} p={p} nu={nu}: setup+FI {time.time()-t0:.0f}s  D0={I.D0:.5g} gc={gc:.4g}', flush=True)
    Nt = I.Nt
    hk = h1 / nu; dh = np.r_[0, np.diff(hk)[:Nt - 1]]
    _, U0 = I.value(0.0); scale = np.abs(U0).max()
    K = lq_gains(disc, M, rho)
    res = []
    for gr in GRS:
        g = gr * gc
        LP = LinearPolicy(I, dh, p, g)
        x, Jlin, info = LP.solve(scale)
        G = LP.unpack(x)
        mc = MonteCarlo(I, hk, p, S=S, seed=12345)
        Jl = mc.run_linear(G, g)
        r = dict(gr=gr, gamma=g, p=p, nu=nu, D0=I.D0, FI_NL=p * FI[gr]['FI_NL'], FI_lin=p * FI[gr]['FI_lin'],
                 lin_exact=Jlin, lin_mc=Jl.mean(), lin_se=Jl.std() / np.sqrt(S))
        tau0 = g / (2 * rho)
        kinds = ['kalman'] if p >= 1 else ['kalman', 'cm']
        for kind in kinds:
            best = None
            for f in TAUF:
                Jf = mc.run_feedback(K, f * tau0, kind, g)
                d = Jf - Jl
                if best is None or Jf.mean() < best[0]:
                    best = (Jf.mean(), f, d.mean(), d.std() / np.sqrt(S))
            r[f'thr_{kind}'] = best
            Jm = mc.run_mpc(g, kind)
            d = Jm - Jl
            r[f'mpc_{kind}'] = (Jm.mean(), d.mean(), d.std() / np.sqrt(S))
        res.append(r)
        msg = ' '.join(f"{k}={v[0]:.5g}(d={v[-2]:+.2e}+-{v[-1]:.1e})" for k, v in r.items() if k.startswith(('thr', 'mpc')))
        print(f"  g/gc={gr}: FI_NL={r['FI_NL']:.5g} FI_lin={r['FI_lin']:.5g} lin={Jlin:.5g} (mc {r['lin_mc']:.5g}+-{r['lin_se']:.1e}) {msg} [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(dict(case=case, p=p, nu=nu, tmax=tmax, rho=rho, gc=gc, D0=I.D0, gs=gs, V=V, res=res),
                open(f'{common.DATA}/pdrag_{case}_p{p:g}_nu{nu:g}.pkl', 'wb'))
    print('done', case, p, nu, time.time() - t0, flush=True)

if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
