"""Reynolds-number dependence of the transient growth and of the optimality ratios."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pickle, sys, time
from orrss import turprof_spectral
from lq import Plant, LQProblem, discounted_costs
import common

TF = np.array([0.5, 1.0, 2.0, 3.0])
TL = np.array([1.0, 2.0, 3.0])

def run(Rt, case, act, N, rhat=1.0):
    t0 = time.time()
    prof = turprof_spectral(N, Rt, 1.0, 'tau')
    if case == 'inner':
        lx, lz = 1000.0 / Rt, 100.0 / Rt
    else:
        lx, lz = 10.0, 3.0
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, Rt=Rt, actuator=act, profile=prof)
    tmax, Gmax, z0, _, _ = pl.transient_growth(tgrid=np.geomspace(1e-5, 20.0, 120))
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    G = []; D = []
    for tf in TF:
        J0, JL, Jo, _ = lq.costs(z0, tf * tmax); G.append(JL / Jo)
    for tl in TL:
        _, a, b, _, _ = discounted_costs(pl.A, pl.B, rho, lq.K, z0, 1.0 / (tl * tmax)); D.append(a / b)
    out = dict(Rt=Rt, case=case, act=act, N=N, tmax=tmax, Gmax=Gmax, Ubp=prof[3], Gamma=np.array(G), Disc=np.array(D), TF=TF, TL=TL)
    pickle.dump(out, open(f'{common.DATA}/re_{case}_{act}_{int(Rt)}_N{N}.pkl', 'wb'))
    print(f'Re={Rt} {case} {act} N={N}: Gmax={Gmax:.4f} tmax={tmax:.5f} (t+={tmax*Rt:.1f}) Gamma={np.round(G,4)} disc={np.round(D,4)} [{time.time()-t0:.0f}s]', flush=True)

if __name__ == '__main__':
    run(float(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
