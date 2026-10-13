"""Scan of penalty rho and horizon T for representative wavenumbers."""
import common
import numpy as np, time, sys, pickle
from multiprocessing import Pool
from lq import Plant, LQProblem, worst_case_ratio

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
TF = np.array([0.25, 0.5, 1, 1.5, 2, 3, 5, 10])

def work(args):
    case, act, rho = args
    lx, lz = CASES[case]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    lq = LQProblem(pl.A, pl.B, rho)
    out = dict(case=case, act=act, rho=rho, tmax=tmax, Gmax=Gmax, Jinf0=float(np.real(z0.conj()@lq.Winf@z0)),
               JinfL=float(np.real(z0.conj()@lq.P@z0)), T=TF*tmax)
    J0, JL, Jo, wc = [], [], [], []
    Zp = pl.Zphys
    for T in TF*tmax:
        a, b, c, cm = lq.costs(z0, T)
        J0.append(a); JL.append(b); Jo.append(c)
        ML = Zp.conj().T @ cm['PL'] @ Zp; Mg = Zp.conj().T @ cm['Gap'] @ Zp
        wc.append((worst_case_ratio(ML, Mg), np.trace(ML).real/np.trace(ML - Mg).real))
    out.update(J0=np.array(J0), JL=np.array(JL), Jo=np.array(Jo), worst=np.array(wc))
    # peak energy under LQR
    t = np.linspace(0, 10*tmax, 400)
    Z, U = lq.lqr_trajectory(z0, t)
    out['GmaxLQR'] = np.max(np.sum(np.abs(Z)**2, 1))
    return out

if __name__ == '__main__':
    rhos = 10.0**np.arange(-7, 3)
    jobs = [(c, a, r) for c in CASES for a in ['wall', 'distributed'] for r in rhos]
    common.profile()
    t0 = time.time()
    with Pool(18) as p:
        res = p.map(work, jobs, chunksize=1)
    pickle.dump(res, open(f'{common.DATA}/scan_rho_T.pkl', 'wb'))
    print('done', time.time()-t0)
    for r in res:
        print(f"{r['case']:5s} {r['act']:11s} rho={r['rho']:.0e} tmax={r['tmax']:.4f} Jinf0={r['Jinf0']:.4f} JinfL={r['JinfL']:.4f} G_LQR={r['GmaxLQR']:.3f}")
        print('     ratio JL/Jo:', np.array2string(r['JL']/r['Jo'], precision=4), ' eff:', np.array2string((r['J0']-r['JL'])/(r['J0']-r['Jo']), precision=3))
        print('     worst:', np.array2string(r['worst'][:,0], precision=3), ' trace:', np.array2string(r['worst'][:,1], precision=4))
