"""Maps over (lambda_x, lambda_z) of the optimality gap of LQR."""
import common
import numpy as np, time, sys, pickle
from multiprocessing import Pool
from lq import Plant, LQProblem, worst_case_ratio, discounted_costs

TF = np.array([0.5, 1.0, 2.0, 3.0, 5.0])        # deterministic horizons T/t_max
TL = np.array([1.0, 2.0, 3.0, 5.0])             # mean random lifetime tau_L/t_max
RHAT = {'wall': [1e-2, 1.0, 1e2], 'distributed': [1e-1, 1.0, 1e1]}

def work(args):
    lx, lz, act = args
    try:
        pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
        tmax, Gmax, z0, _, _ = pl.transient_growth(tgrid=np.geomspace(1e-4, 20.0, 90))
    except Exception as e:
        return dict(lx=lx, lz=lz, act=act, err=str(e))
    out = dict(lx=lx, lz=lz, act=act, tmax=tmax, Gmax=Gmax, res=[])
    if Gmax < 1.05:
        return out
    Zp = pl.Zphys
    for rh in RHAT[act]:
        rho = rh * (tmax**2 if act == 'distributed' else tmax)
        try:
            lq = LQProblem(pl.A, pl.B, rho)
        except Exception as e:
            out['res'].append(dict(rhat=rh, err=str(e))); continue
        r = dict(rhat=rh, rho=rho, Jinf0=float(np.real(z0.conj()@lq.Winf@z0)), JinfL=float(np.real(z0.conj()@lq.P@z0)))
        J0 = []; JL = []; Jo = []; wc = []; tr = []
        for T in TF*tmax:
            a, b, c, cm = lq.costs(z0, T)
            J0.append(a); JL.append(b); Jo.append(c)
            ML = Zp.conj().T @ cm['PL'] @ Zp; Mg = Zp.conj().T @ cm['Gap'] @ Zp
            wc.append(worst_case_ratio(ML, Mg)); tr.append(np.trace(ML).real/np.trace(ML-Mg).real)
        r.update(J0=np.array(J0), JL=np.array(JL), Jo=np.array(Jo), worst=np.array(wc), trace=np.array(tr))
        D0 = []; DL = []; Do = []
        for tl in TL*tmax:
            a, b, c, Kmu, _ = discounted_costs(pl.A, pl.B, rho, lq.K, z0, 1.0/tl)
            D0.append(a); DL.append(b); Do.append(c)
        r.update(D0=np.array(D0), DL=np.array(DL), Do=np.array(Do))
        out['res'].append(r)
    return out

if __name__ == '__main__':
    nx, nz = int(sys.argv[1]), int(sys.argv[2])
    LX = np.geomspace(0.1, 100.0, nx)
    LZ = np.geomspace(0.02, 10.0, nz)
    common.profile()
    jobs = [(lx, lz, act) for act in ['wall', 'distributed'] for lx in LX for lz in LZ]
    # interleave expensive/cheap jobs
    t0 = time.time()
    with Pool(18) as p:
        res = []
        for k, r in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            res.append(r)
            if k % 50 == 0:
                print(k, len(jobs), f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(LX=LX, LZ=LZ, TF=TF, TL=TL, RHAT=RHAT, res=res), open(f'{common.DATA}/maps_{nx}x{nz}.pkl', 'wb'))
    print('done', time.time()-t0)
