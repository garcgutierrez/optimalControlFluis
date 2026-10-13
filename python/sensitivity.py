"""Sensitivity of the optimality ratio to resolution, actuator time constant and eddy viscosity."""
import common
import numpy as np, pickle, time
from multiprocessing import Pool
from lq import Plant, LQProblem
from orrss import turprof_spectral

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
TF = [1.0, 2.0, 3.0]

def work(a):
    case, act, N, rbcfac, ef = a
    lx, lz = CASES[case]
    prof = turprof_spectral(N, common.RT, ef, 'tau')
    Rbc = None if rbcfac == 1 else rbcfac * common.RT * prof[3]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator=act, profile=prof, Rbc=Rbc)
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = 1.0 * (tmax**2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    out = []
    for f in TF:
        J0, JL, Jo, _ = lq.costs(z0, f*tmax)
        out.append((J0, JL, Jo))
    return dict(case=case, act=act, N=N, rbc=rbcfac, ef=ef, tmax=tmax, Gmax=Gmax, J=np.array(out))

if __name__ == '__main__':
    jobs = []
    for case in CASES:
        for act in ['wall', 'distributed']:
            for N in [96, 128, 192]:
                jobs.append((case, act, N, 1, 1.0))
            jobs.append((case, act, 128, 1, 0.5))
            if act == 'wall':
                jobs += [(case, act, 128, 0.2, 1.0), (case, act, 128, 5.0, 1.0)]
    with Pool(4) as p:
        res = p.map(work, jobs, chunksize=1)
    pickle.dump(res, open(f'{common.DATA}/sensitivity.pkl', 'wb'))
    for r in res:
        J = r['J']
        print(f"{r['case']:5s} {r['act']:11s} N={r['N']} Rbc x{r['rbc']} ef={r['ef']}: Gmax={r['Gmax']:.4f} tmax={r['tmax']:.5f} "
              f"J0={np.array2string(J[:,0],precision=4)} JL={np.array2string(J[:,1],precision=4)} Jo={np.array2string(J[:,2],precision=4)} ratio={np.array2string(J[:,1]/J[:,2],precision=4)}")
