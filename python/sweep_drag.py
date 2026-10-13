"""Drag objective with amplitude-proportional actuation cost: nonlinear optimum versus the best
linear feedback, for bursts of random amplitude a ~ CN(0,1)."""
import common
import numpy as np, time, pickle, sys
from lq import Plant, LQProblem, Discretised, saturated_lqr
from drag import drag_matrix, DragProblem, expected_costs

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}

def run(case, act, TF, rhat=1.0, Nt=300):
    lx, lz = CASES[case]
    t0 = time.time()
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    T = TF * tmax
    disc = Discretised(pl.A, pl.B, T, Nt)
    MD = drag_matrix(pl)
    P = DragProblem(disc, z0, MD, rho, 'input' if act == 'wall' else 'vector')
    lmax, lmin = P.hessian_extremes()
    L = 1.05 * lmax
    D0 = P.smooth(np.zeros((Nt, disc.m), complex))[0]
    gc = P.threshold()
    print(f'{case} {act} {TF}: D0={D0:.5g} Hessian [{lmin:.3g}, {lmax:.3g}] gamma_c={gc:.4g} [{time.time()-t0:.0f}s]', flush=True)
    if lmin <= 0:
        print('  WARNING: not convex')
    # value function V(g) by continuation from g = gamma_c (U = 0) downwards
    gs = np.r_[gc * np.geomspace(1.0, 1e-3, 31), 0.0]
    U = np.zeros((Nt, disc.m), complex)
    V, Qs, L1s, Us = [], [], [], {}
    for g in gs:
        U, Q, l1, it = P.solve(g, L, U0=U)
        V.append(Q + g * l1); Qs.append(Q); L1s.append(l1)
        if np.isclose(g, gc * 0.3, rtol=0.2) or np.isclose(g, gc * 0.03, rtol=0.2) or g == 0:
            Us[g] = U.copy()
        print(f'  g/gc={g/gc:.3e}: V={V[-1]:.6g} Q={Q:.6g} l1={l1:.4g} its={it} [{time.time()-t0:.0f}s]', flush=True)
    V = np.array(V); Qs = np.array(Qs); L1s = np.array(L1s)
    # energy-optimal LQR (from the paper) evaluated on the drag objective
    lq = LQProblem(pl.A, pl.B, rho)
    UL, _, _ = saturated_lqr(disc, z0, lq.K, rho, 1e30)
    QL = P.smooth(UL)[0]; l1L = P.l1(UL)
    out = dict(case=case, act=act, TF=TF, tmax=tmax, rho=rho, D0=D0, gc=gc, gs=gs, V=V, Q=Qs, L1=L1s, Us=Us,
               QLQR=QL, l1LQR=l1L, ULQR=UL, t=np.arange(Nt) * disc.h + 0.5 * disc.h, lmin=lmin, lmax=lmax)
    pickle.dump(out, open(f'{common.DATA}/drag_{case}_{act}_T{TF:g}.pkl', 'wb'))
    print('done', case, act, TF, time.time()-t0, flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], float(sys.argv[3]))
