"""Data for the representative cases (verification, time histories, flow fields,
bounded actuation).  Results are stored in data/cases.pkl."""
import common
import numpy as np, time, pickle, sys
import scipy.linalg as sla
from lq import (Plant, LQProblem, Discretised, direct_optimisation, discrete_riccati_optimum,
                constrained_optimum, saturated_lqr, lipschitz, discounted_costs)

CASES = {'inner': (0.5, 0.05), 'outer': (10.0, 3.0)}
RHAT = {'wall': 1.0, 'distributed': 1.0}


def fields(pl, Z, x, nz_keep=None):
    """Streamwise and wall-normal velocity in the (x, y) plane z = 0 for each state."""
    mod = pl.mod
    out_u, out_v = [], []
    ph = np.exp(1j * pl.alp * x)
    for z in Z:
        q = pl.to_q(z)
        u, v, w = mod.velocities(q)
        out_u.append(np.real(np.outer(u, ph)))
        out_v.append(np.real(np.outer(v, ph)))
    return np.array(out_u), np.array(out_v)


def energy_profile(pl, z):
    q = pl.to_q(z)
    u, v, w = pl.mod.velocities(q)
    return np.abs(u) ** 2, np.abs(v) ** 2, np.abs(w) ** 2


def run_case(case, act, rhat, verification=False, bounded=False):
    lx, lz = CASES[case]
    t0 = time.time()
    pl = Plant(2 * np.pi / lx, 2 * np.pi / lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, tg, g = pl.transient_growth()
    rho = rhat * (tmax ** 2 if act == 'distributed' else tmax)
    lq = LQProblem(pl.A, pl.B, rho)
    R = dict(case=case, act=act, lx=lx, lz=lz, tmax=tmax, Gmax=Gmax, rho=rho, rhat=rhat,
             y=pl.mod.y, tg=tg, g=g)
    # time histories for T = tmax, 2 tmax, 3 tmax
    hist = {}
    for TF in [1.0, 2.0, 3.0]:
        T = TF * tmax
        t = np.linspace(0, 1.5 * T, 601)
        tt = t[t <= T]
        Z0 = lq.uncontrolled_trajectory(z0, t)
        ZL, UL = lq.lqr_trajectory(z0, t)
        Zo, Uo = lq.optimal_trajectory(z0, T, tt)
        J0, JL, Jo, cm = lq.costs(z0, T)
        hist[TF] = dict(t=t, tt=tt, E0=np.sum(abs(Z0) ** 2, 1), EL=np.sum(abs(ZL) ** 2, 1),
                        UL=np.abs(UL), Eo=np.sum(abs(Zo) ** 2, 1), Uo=np.abs(Uo),
                        J0=J0, JL=JL, Jo=Jo)
        # energy profiles at t = T
        kT = np.argmin(abs(t - T))
        hist[TF]['prof0'] = energy_profile(pl, Z0[kT])
        hist[TF]['profL'] = energy_profile(pl, ZL[kT])
        hist[TF]['profo'] = energy_profile(pl, Zo[-1])
        # flow fields at t = 0, T/2, T
        x = np.linspace(0, lx, 129)
        ks = [0, np.argmin(abs(tt - T / 2)), len(tt) - 1]
        kL = [0, np.argmin(abs(t - T / 2)), kT]
        hist[TF]['x'] = x
        hist[TF]['fu0'], hist[TF]['fv0'] = fields(pl, Z0[kL], x)
        hist[TF]['fuL'], hist[TF]['fvL'] = fields(pl, ZL[kL], x)
        hist[TF]['fuo'], hist[TF]['fvo'] = fields(pl, Zo[ks], x)
        print(f'  {case} {act} T/tmax={TF}: J0={J0:.4f} JL={JL:.4f} Jo={Jo:.4f} ratio={JL/Jo:.4f}', flush=True)
    R['hist'] = hist
    # scans in T/tmax and rhat (dense)
    TFs = np.geomspace(0.1, 10, 41)
    rhats = np.geomspace(1e-3, 1e3, 13) if act == 'wall' else np.geomspace(1e-2, 1e2, 13)
    scan = np.zeros((len(rhats), len(TFs), 3))
    disc = np.zeros((len(rhats), len(TFs), 3))
    for i, rh in enumerate(rhats):
        rr = rh * (tmax ** 2 if act == 'distributed' else tmax)
        try:
            lqi = LQProblem(pl.A, pl.B, rr)
        except Exception as e:
            scan[i] = np.nan; disc[i] = np.nan; continue
        for j, TF in enumerate(TFs):
            a, b, c, _ = lqi.costs(z0, TF * tmax)
            scan[i, j] = a, b, c
            a, b, c, _, _ = discounted_costs(pl.A, pl.B, rr, lqi.K, z0, 1.0 / (TF * tmax))
            disc[i, j] = a, b, c
    R.update(TFs=TFs, rhats=rhats, scan=scan, disc=disc)
    print(f'  scans done {time.time()-t0:.0f}s', flush=True)
    if verification:
        T = tmax
        J0, JL, Jo, _ = lq.costs(z0, T)
        ver = dict(T=T, J0=J0, JL=JL, Jo=Jo, Nt=[], Jd=[], Jdo=[], JLd=[], its=[])
        for Nt in [25, 50, 100, 200, 400, 800]:
            dsc = Discretised(pl.A, pl.B, T, Nt)
            Ud, Jd, Zd = discrete_riccati_optimum(dsc, z0, rho)
            Uo, Jdo, h, res = direct_optimisation(dsc, z0, rho, maxiter=5000, gtol=1e-12, history=True)
            Us, JLd, Zs = saturated_lqr(dsc, z0, lq.K, rho, 1e30)
            ver['Nt'].append(Nt); ver['Jd'].append(Jd); ver['Jdo'].append(Jdo); ver['JLd'].append(JLd)
            ver['its'].append(res.nit)
            if Nt == 200:
                ver['hist'] = np.array(h)
                ver['t_d'] = np.arange(Nt) * dsc.h + 0.5 * dsc.h
                ver['U_direct'] = np.abs(Uo)
                ver['U_ric'] = np.abs(Ud)
            print(f'   Nt={Nt}: Jd={Jd:.7f} Jdirect={Jdo:.7f} its={res.nit} JLQRd={JLd:.7f}', flush=True)
        R['ver'] = ver
    if bounded:
        B = {}
        for TF in [1.0, 3.0]:
            T = TF * tmax
            dsc = Discretised(pl.A, pl.B, T, 300)
            Ud, Jd, Zd = discrete_riccati_optimum(dsc, z0, rho)
            Ul, Jl, Zl = saturated_lqr(dsc, z0, lq.K, rho, 1e30)
            uref = np.abs(Ul).max()
            L = lipschitz(dsc, z0, rho)
            fr = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05])
            Jc = []; Js = []; Jdet = []; Uc_store = {}; Us_store = {}
            U0 = Ud.copy()
            for f in fr:
                um = f * uref
                Uc, J, it = constrained_optimum(dsc, z0, rho, um, U0=U0, L=L, maxiter=40000, tol=1e-11)
                U0 = Uc
                Us, Jsat, Zs = saturated_lqr(dsc, z0, lq.K, rho, um)
                # linear law detuned (larger penalty) so that it never exceeds the bound
                lo, hi = np.log(rho), np.log(rho) + 30
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lqm = LQProblem(pl.A, pl.B, np.exp(mid))
                    Um, _, _ = saturated_lqr(dsc, z0, lqm.K, rho, 1e30)
                    if np.abs(Um).max() > um:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 1e-3:
                        break
                lqm = LQProblem(pl.A, pl.B, np.exp(hi))
                Um, Jm, _ = saturated_lqr(dsc, z0, lqm.K, rho, 1e30)
                Jc.append(J); Js.append(Jsat); Jdet.append(Jm)
                Uc_store[f] = np.abs(Uc); Us_store[f] = np.abs(Us)
                print(f'   bounded T/tmax={TF} f={f}: Jc={J:.5f} Jsat={Jsat:.5f} Jdetuned={Jm:.5f} (its {it})', flush=True)
            B[TF] = dict(fr=fr, uref=uref, Jd=Jd, Jl=Jl, J0=lq.costs(z0, T)[0], Jc=np.array(Jc), Js=np.array(Js),
                         Jdet=np.array(Jdet), Uc=Uc_store, Us=Us_store, td=np.arange(300) * dsc.h + 0.5 * dsc.h)
        R['bounded'] = B
    print(f' {case} {act} done {time.time()-t0:.0f}s', flush=True)
    return R


if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    common.profile()
    out = {}
    for case in CASES:
        for act in ['wall', 'distributed']:
            key = f'{case}_{act}'
            if 'all' not in which and key not in which:
                continue
            out[key] = run_case(case, act, RHAT[act], verification=(key == 'outer_wall'),
                                bounded=(act == 'wall' or key == 'outer_distributed'))
            pickle.dump(out[key], open(f'{common.DATA}/case_{key}.pkl', 'wb'))
