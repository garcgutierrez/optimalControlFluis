"""Pulsed wall transpiration versus linear control (linear model, quadratic cost).

A K-pulse control is phi(t) = sum_k a_k 1[t_k, t_k + w), with free complex amplitudes a_k at both walls.
For given pulse times the cost is quadratic in the amplitudes and is minimised exactly; the times are
optimised by greedy insertion followed by sweeps of single-pulse relocation."""
import common
import numpy as np, pickle, sys, time
from lq import Plant, LQProblem, Discretised
from pdrag import InputSpace

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}


def setup(case, TF, Nt=200, rhat=1.0):
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator='wall', profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * tmax; T = TF * tmax
    disc = Discretised(pl.A, pl.B, T, Nt)
    I = InputSpace(disc, np.eye(pl.n), rho, z0)          # energy objective: M = I
    J0, JL, Jo, _ = LQProblem(pl.A, pl.B, rho).costs(z0, T)
    return I, dict(J0=J0, JL=JL, Jopt=Jo, tmax=tmax)


class Pulses:
    def __init__(self, I, W):
        self.H, self.b, self.c0 = I.H, I.b0, I.D0
        self.Nt, self.m, self.W = I.Nt, I.m, W
        self.starts = list(range(0, self.Nt - W + 1))

    def idx(self, s, j):
        return np.arange(s, s + self.W) * self.m + j

    def cost(self, starts):
        cols = [self.idx(s, j) for s in starts for j in range(self.m)]
        n = len(cols)
        PHP = np.empty((n, n), complex); Pb = np.empty(n, complex)
        for a in range(n):
            Pb[a] = self.b[cols[a]].sum()
            for c in range(a, n):
                PHP[a, c] = self.H[np.ix_(cols[a], cols[c])].sum()
                PHP[c, a] = np.conj(PHP[a, c])
        x = np.linalg.solve(PHP, -Pb)
        return float(self.c0 + np.real(np.vdot(Pb, x))), x

    def free(self, s, others):
        return all(abs(s - o) >= self.W for o in others)

    def optimise(self, Kmax):
        chosen, out = [], {}
        for K in range(1, Kmax + 1):
            best = None
            for s in self.starts:
                if self.free(s, chosen):
                    J, _ = self.cost(chosen + [s])
                    if best is None or J < best[0]:
                        best = (J, s)
            chosen.append(best[1])
            improved = True
            while improved:                       # relocate one pulse at a time
                improved = False
                for i in range(len(chosen)):
                    others = chosen[:i] + chosen[i + 1:]
                    Jc, _ = self.cost(chosen)
                    for s in self.starts:
                        if self.free(s, others):
                            J, _ = self.cost(others + [s])
                            if J < Jc - 1e-14 * abs(Jc):
                                chosen = others + [s]; chosen.sort(); Jc = J; improved = True
                                others = [c for c in chosen if c != s]
            J, x = self.cost(sorted(chosen))
            out[K] = dict(J=J, starts=sorted(chosen), amp=x)
        return out


def run(case, TF, Ws=(1, 4, 10), Kmax=10):
    t0 = time.time()
    I, ref = setup(case, TF)
    # all intervals free: exact discrete optimum
    x = np.linalg.solve(I.H, -I.b0); Jd = float(I.D0 + np.real(np.vdot(I.b0, x)))
    res = dict(case=case, TF=TF, ref=ref, Jdisc=Jd, Nt=I.Nt, pulses={})
    print(f"{case} T={TF}tmax: J0/J*={ref['J0']/ref['Jopt']:.4f} LQR/J*={ref['JL']/ref['Jopt']:.4f} discrete opt/J*={Jd/ref['Jopt']:.5f} [{time.time()-t0:.0f}s]", flush=True)
    for W in Ws:
        P = Pulses(I, W)
        out = P.optimise(Kmax)
        res['pulses'][W] = out
        s = ' '.join(f"K={K}:{out[K]['J']/ref['Jopt']:.4f}" for K in out)
        print(f"   width {W*TF/I.Nt:.3f} tmax: {s} [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(res, open(f'{common.DATA}/pulses_{case}_T{TF:g}.pkl', 'wb'))


if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]))
