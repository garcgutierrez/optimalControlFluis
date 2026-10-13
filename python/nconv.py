"""Resolution study of the transient growth and of the optimality ratio (inner case, Re_tau=2000)."""
import common, numpy as np, sys, pickle, time
from lq import Plant, LQProblem
case = sys.argv[1]
lx, lz = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}[case]
out = []
for N in range(96, 272, 16):
    t0 = time.time()
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator='wall', profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    lq = LQProblem(pl.A, pl.B, tmax)
    J0, JL, Jo, _ = lq.costs(z0, tmax)
    out.append((N, Gmax, tmax, J0, JL, Jo))
    print(f'{case} N={N}: Gmax={Gmax:.4f} tmax={tmax:.5f} J0={J0:.5g} JL={JL:.5g} Jopt={Jo:.5g} Gamma={JL/Jo:.4f} [{time.time()-t0:.0f}s]', flush=True)
pickle.dump(out, open(f'{common.DATA}/nconv_{case}.pkl', 'wb'))
