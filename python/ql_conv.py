"""Convergence of the quasi-linear model with the time step and the resolution (uncontrolled and LQR)."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import common
import numpy as np, sys
import jax, jax.numpy as jnp
from lq import Plant, LQProblem
from qlmodel import QLModel
from ql_study import CASES

case, act, TF = sys.argv[1], sys.argv[2], float(sys.argv[3])
for N in [128, 192]:
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=N, actuator=act, profile=common.profile(N))
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = tmax**2 if act == 'distributed' else tmax
    lq = LQProblem(pl.A, pl.B, rho); K = jnp.asarray(lq.K)
    for Nt in ([150, 300, 600] if N == 128 else [300]):
        ql = QLModel(pl); ql.build(TF * tmax / Nt)
        f0 = jax.jit(lambda z: ql.simulate(z, jnp.zeros((Nt, pl.m), complex), rho)[0])
        fK = jax.jit(lambda z: ql.simulate(z, None, rho, feedback=(K, Nt))[0])
        s = []
        for eps in [0.1, 1.0, 2.0]:
            s.append(f'eps={eps}: J0/e2={float(f0(eps*z0))/eps**2:.6f} LQR/e2={float(fK(eps*z0))/eps**2:.6f}')
        print(f'{case} {act} T={TF} N={N} Nt={Nt}: ' + ' | '.join(s), flush=True)
