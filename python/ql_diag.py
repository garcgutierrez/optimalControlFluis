"""Diagnostics of the quasi-linear model: amplitude of the bursts in physical units and size of the
mean-flow distortion of the uncontrolled bursts."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import common
import numpy as np, pickle, sys
import jax, jax.numpy as jnp
import scipy.linalg as sla
from lq import Plant
from qlmodel import QLModel
from ql_study import CASES, EPS

def run(case, TF=2.0, Nt=300, N=128):
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator='wall', profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    T = TF * tmax; h = T / Nt
    ql = QLModel(pl); ql.build(h)
    a, b, k2 = ql.a, ql.b, ql.k2
    def uvel(z):
        v = ql.Vn @ z; dv = ql.dVn @ z; eta = ql.En @ z
        return 1j * (a * dv - b * eta) / k2, v
    zt = sla.expm(pl.A * tmax) @ z0
    u, v = uvel(zt)
    y = ql.y
    urms = np.sqrt(2) * np.abs(u); ymax = y[np.argmax(urms)]
    print(f'{case}: linear burst, eps=1 at t_max: max u_rms={urms.max():.3f} u_tau at y+={(1-abs(ymax))*common.RT:.0f}; '
          f'max v_rms={np.sqrt(2)*np.abs(v).max():.3f}', flush=True)
    step = jax.jit(ql.step)
    out = []
    for eps in EPS:
        st = (jnp.asarray(eps * z0), jnp.zeros(ql.nint)); U1m = 0.0; Ef = []
        for k in range(Nt):
            st = step(st, jnp.zeros(pl.m, complex))
            U1m = max(U1m, float(jnp.abs(st[1]).max())); Ef.append(float(jnp.real(jnp.vdot(st[0], st[0]))))
        Gnl = max(Ef) / eps**2
        out.append((eps, U1m, Gnl))
        print(f'  eps={eps}: max|U1|={U1m:.3f} u_tau  max E/E0={Gnl:.3f} (linear {Gmax:.3f})', flush=True)
    pickle.dump(dict(case=case, urms=urms, y=y, tmax=tmax, Gmax=Gmax, out=out), open(f'{common.DATA}/qldiag_{case}.pkl', 'wb'))

if __name__ == '__main__':
    run(sys.argv[1])
