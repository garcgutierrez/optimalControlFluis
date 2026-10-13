"""Drag objective from noisy wall-pressure measurements: direct search of a nonlinear output-feedback
policy by sample-average approximation (SAA).

Policy class (contains the best linear output feedback, c = 0, tau = 0):
    u_k = soft( sum_{l<=k} G[k,l] dm_l + c_k betahat_k , tau_k )
where dm_l are the increments of the matched-filter statistic, betahat_k is the conditional mean of the
burst amplitude (the detector of the intermittent case) and soft() is a complex soft threshold (dead zone).
The parameters are optimised with L-BFGS on the average cost of S_train Monte Carlo bursts (smoothed
dead zone and l1 term, with continuation), starting from the best linear policy, and the result is
evaluated exactly (no smoothing) on the independent samples used for all the other policies."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import common
import numpy as np, pickle, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.optimize import minimize
from pdrag import LinearPolicy, MonteCarlo, csoft
from sweep_pdrag import setup
GRS = [0.3, 0.7]


class PolicySearch:
    def __init__(self, I, LP, p):
        self.I, self.LP, self.p = I, LP, p
        Nt, m = I.Nt, I.m
        self.Nt, self.m = Nt, m
        self.kk, self.ll = np.nonzero(LP.mask)
        self.nG = len(self.kk)
        self.H, self.b0 = jnp.asarray(I.H), jnp.asarray(I.b0)

    def data(self, mc):
        dm = np.diff(mc.m, axis=0)                                   # dm[l-1] available at t_l
        bh = np.array([mc.estimate(k, 'cm') for k in range(self.Nt)])  # (Nt, S)
        return jnp.asarray(dm), jnp.asarray(bh), jnp.asarray(mc.beta)

    def unpack(self, x):
        Nt, m, nG = self.Nt, self.m, self.nG
        o = 0
        Gv = x[o:o + nG * m] + 1j * x[o + nG * m:o + 2 * nG * m]; o += 2 * nG * m
        c = (x[o:o + Nt * m] + 1j * x[o + Nt * m:o + 2 * Nt * m]).reshape(Nt, m); o += 2 * Nt * m
        th = x[o:o + Nt * m].reshape(Nt, m)
        G = jnp.zeros((Nt, Nt, m), complex).at[self.kk, self.ll].set(Gv.reshape(nG, m))
        return G, c, th

    def pack(self, G, c, th):
        Gv = np.asarray(G)[self.kk, self.ll].ravel()
        c = np.asarray(c).ravel(); th = np.asarray(th).ravel()
        return np.concatenate([Gv.real, Gv.imag, c.real, c.imag, th])

    def controls(self, x, dm, bh, scale, eps, exact=False):
        G, c, th = self.unpack(x)
        pre = jnp.einsum('klj,ls->kjs', G[:, 1:, :], dm[:self.Nt - 1]) + c[:, :, None] * bh[:, None, :]
        tau = scale * jax.nn.softplus(th)[:, :, None]
        if exact:
            a = jnp.abs(pre)
            U = pre * jnp.maximum(0.0, 1.0 - tau / jnp.maximum(a, 1e-300))
        else:
            e = eps * scale
            r = jnp.sqrt(jnp.abs(pre) ** 2 + e ** 2)
            f = 1.0 - tau / r
            U = pre * 0.5 * (f + jnp.sqrt(f ** 2 + eps ** 2))
        return U.reshape(self.Nt * self.m, -1)

    def cost(self, x, dm, bh, beta, gamma, scale, eps, exact=False):
        I = self.I
        U = self.controls(x, dm, bh, scale, eps, exact)
        HU = self.H @ U
        J = (jnp.abs(beta) ** 2 * I.D0 + 2 * jnp.real(beta * (jnp.conj(U) * self.b0[:, None]).sum(0))
             + jnp.real((jnp.conj(U) * HU).sum(0)))
        if exact:
            l1 = jnp.abs(U).sum(0)
        else:
            e = eps * scale
            l1 = (jnp.sqrt(jnp.abs(U) ** 2 + e ** 2) - e).sum(0)
        return J + gamma * I.h * l1

    def train(self, x0, mc, gamma, scale, eps_list=(1e-2, 1e-3), maxiter=1500):
        dm, bh, beta = self.data(mc)
        x = x0.copy(); info = []
        for eps in eps_list:
            f = jax.jit(jax.value_and_grad(lambda z: jnp.mean(self.cost(z, dm, bh, beta, gamma, scale, eps))))
            def fg(z):
                v, g = f(jnp.asarray(z)); return float(v), np.asarray(g, dtype=float)
            r = minimize(fg, x, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxcor=30, ftol=1e-13, gtol=1e-10))
            x = r.x; info.append((eps, r.nit, r.fun))
        return x, info

    def evaluate(self, x, mc, gamma, scale):
        dm, bh, beta = self.data(mc)
        return np.asarray(self.cost(jnp.asarray(x), dm, bh, beta, gamma, scale, 0.0, exact=True))


def run(case, p, nu, S=4000, Strain=8000):
    t0 = time.time()
    pl, tmax, z0, rho, disc, M, I, h1 = setup(case)
    gc = I.threshold()
    Nt = I.Nt
    hk = h1 / nu; dh = np.r_[0, np.diff(hk)[:Nt - 1]]
    _, U0 = I.value(0.0); scale = np.abs(U0).max()
    print(f'{case} p={p} nu={nu}: setup {time.time()-t0:.0f}s', flush=True)
    mc_test = MonteCarlo(I, hk, p, S=S, seed=12345)           # same samples as sweep_pdrag
    mc_train = MonteCarlo(I, hk, p, S=Strain, seed=777)
    out = []
    for gr in GRS:
        g = gr * gc
        LP = LinearPolicy(I, dh, p, g)
        xl, Jlin, _ = LP.solve(scale)
        Glin = LP.unpack(xl)
        Jl = mc_test.run_linear(Glin, g)
        PS = PolicySearch(I, LP, p)
        x0 = PS.pack(Glin, np.zeros((Nt, I.m)), np.full((Nt, I.m), -6.0))
        # sanity: the initial policy reproduces the best linear one
        J0 = PS.evaluate(x0, mc_test, g, scale)
        x, info = PS.train(x0, mc_train, g, scale)
        Jp = PS.evaluate(x, mc_test, g, scale)
        Jtr = PS.evaluate(x, mc_train, g, scale).mean()
        d = Jp - Jl
        rec = dict(gr=gr, gamma=g, lin_exact=Jlin, lin_mc=Jl.mean(), init_diff=float(np.abs(J0 - Jl).max()),
                   ps=(Jp.mean(), d.mean(), d.std() / np.sqrt(S)), ps_train=Jtr, info=info)
        out.append(rec)
        print(f'  g/gc={gr}: lin={Jlin:.6g} (mc {Jl.mean():.6g}) policy search: test {Jp.mean():.6g} '
              f'(d={d.mean():+.3e}+-{d.std()/np.sqrt(S):.1e}) train {Jtr:.6g} init-check {rec["init_diff"]:.1e} '
              f'{[(e, n) for e, n, _ in info]} [{time.time()-t0:.0f}s]', flush=True)
    pickle.dump(dict(case=case, p=p, nu=nu, res=out), open(f'{common.DATA}/pdragps_{case}_p{p:g}_nu{nu:g}.pkl', 'wb'))
    print('done', case, p, nu, time.time() - t0, flush=True)


if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
