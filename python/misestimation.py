"""Robustness to the lifetime assumed in the design (full information, energy objective).
Deterministic: policy optimal for T_d applied to a burst that lives T (zero control after T_d).
Random: discounted LQR designed for mean lifetime tau_d applied to exponential lifetimes of mean tau_L."""
import common
import numpy as np, pickle, sys, time
import scipy.linalg as sla
from lq import Plant, LQProblem, discounted_costs, care, lyap
from orrss import herm

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
TD = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf])
TA = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])

def run(case, act, rhat=1.0):
    t0 = time.time()
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = rhat * (tmax**2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    n = lq.n; I = np.eye(n)
    P, Ac, Ginf, Winf = lq.P, lq.Ac, lq.Ginf, lq.Winf
    def Popt(tau):
        Ec = sla.expm(Ac * tau); GT = herm(Ginf - Ec @ Ginf @ Ec.conj().T)
        PL = herm(P - Ec.conj().T @ P @ Ec)
        X = np.linalg.solve(I - GT @ P, GT @ P @ Ec)
        return herm(PL - Ec.conj().T @ P @ X)
    def W0(tau):
        E0 = sla.expm(lq.A * tau); return herm(Winf - E0.conj().T @ Winf @ E0)
    q = lambda M, z: float(np.real(z.conj() @ M @ z))
    Jopt = {ta: q(Popt(ta * tmax), z0) for ta in TA}
    R = np.zeros((len(TD), len(TA)))
    for i, td in enumerate(TD):
        for j, ta in enumerate(TA):
            T = ta * tmax
            if np.isinf(td):
                R[i, j] = q(lq.cost_matrices(T)['PL'], z0)
                continue
            Td = td * tmax
            if T <= Td:
                # cost over [0,T] of the T_d-optimal policy: z0^H P_Td(0) z0 - z(T)^H P_Td(T) z(T)
                Z, _ = lq.optimal_trajectory(z0, Td, np.array([T]))
                R[i, j] = q(Popt(Td), z0) - q(Popt(Td - T), Z[0])
            else:
                Z, _ = lq.optimal_trajectory(z0, Td, np.array([Td]))
                R[i, j] = q(Popt(Td), z0) + q(W0(T - Td), Z[0])
        print(f'  {case} {act} T_d={td}: ratios {np.round(R[i] / np.array([Jopt[t] for t in TA]), 4)} [{time.time()-t0:.0f}s]', flush=True)
    Jo = np.array([Jopt[t] for t in TA])
    # random lifetimes: design mean tau_d, actual mean tau_L
    TL = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    TDR = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf])
    Rr = np.zeros((len(TDR), len(TL))); Jr = np.zeros(len(TL))
    Kd = {}
    for i, tdr in enumerate(TDR):
        if np.isinf(tdr):
            Kd[i] = lq.K
        else:
            Pm = care(pl.A - 0.5 / (tdr * tmax) * I, pl.B, rho); Kd[i] = pl.B.conj().T @ Pm / rho
    for j, tl in enumerate(TL):
        mu = 1.0 / (tl * tmax)
        for i in range(len(TDR)):
            _, JL, Jmu, _, _ = discounted_costs(pl.A, pl.B, rho, Kd[i], z0, mu)
            Rr[i, j] = JL; Jr[j] = Jmu
    out = dict(case=case, act=act, tmax=tmax, TD=TD, TA=TA, R=R, Jopt=Jo, TL=TL, TDR=TDR, Rr=Rr, Jr=Jr)
    pickle.dump(out, open(f'{common.DATA}/mis_{case}_{act}.pkl', 'wb'))
    print('done', case, act, time.time() - t0, flush=True)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
