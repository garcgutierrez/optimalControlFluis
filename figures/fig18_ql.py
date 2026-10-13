"""Figure 18 (fig:ql): quasi-linear model with mean-flow distortion, rho_hat = 1.  Costs
normalised with eps^2 and with the optimal cost of the linear model, J*_lin, versus the
amplitude of the burst (peak r.m.s. velocity).  Uncontrolled, LQR, finite-horizon linear
feedback (designed and re-tuned) and nonlinear optimum.  Rows: T = t_max, 2 t_max.

Data  : ql_<case>_<actuator>_T<T>*.pkl, qltuned_<case>_<actuator>_T<T>*.pkl
        (python ql_study.py <case> <actuator> <T> full|tuned <eps list>), qldiag_<case>.pkl
        (python ql_diag.py <case>); J*_lin computed on the fly (lq.LQProblem)
Output: fig18_ql.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import D, C_UNC, C_LQR, C_OPT, C_3, LS_UNC, LS_LQR, LS_3, W1, CASEMARK, COMBOS, label, save, load
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
import common
from lq import Plant, LQProblem

LIFETIMES = [1.0, 2.0]      # T/t_max (rows)
SHADE_FROM = 3.0            # u'_max/u_tau above which the panel is shaded


def load_ql(case, act, TF):
    recs = []
    for f in sorted(glob.glob(os.path.join(D, f'ql_{case}_{act}_T{TF:g}*.pkl'))):
        recs += load(f)['res']
    if not recs:
        return None
    recs = sorted(recs, key=lambda r: r['eps'])
    out = {k: np.array([r[k] for r in recs]) for k in ['eps', 'J0', 'LQR', 'LTV', 'open', 'NLopt']}
    for kind, key in [('JL', 'LQRt'), ('JT', 'LTVt')]:
        vals = np.full(len(recs), np.nan)
        for f in glob.glob(os.path.join(D, f'qltuned_{case}_{act}_T{TF:g}*.pkl')):
            t = load(f)
            for j, e in enumerate(t['EPS']):
                i = np.where(np.isclose(out['eps'], e))[0]
                if len(i):
                    vals[i[0]] = np.nanmin([vals[i[0]], np.nanmin(t[kind][:, j])])
        out[key] = vals
    return out


def ql_linear_optimum(case, act, TF):
    lx, lz = CASEMARK[case][1:]
    pl = Plant(2*np.pi/lx, 2*np.pi/lz, N=common.NCHEB, actuator=act, profile=common.profile())
    tmax, Gmax, z0, _, _ = pl.transient_growth()
    rho = tmax**2 if act == 'distributed' else tmax
    J0, JL, Jo, _ = LQProblem(pl.A, pl.B, rho).costs(z0, TF * tmax)
    return Jo


def main():
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.6), sharey='row')
    umax = {case: float(load(f'qldiag_{case}.pkl')['urms'].max()) for case in ['inner', 'outer']}
    for r, TF in enumerate(LIFETIMES):
        for c, (case, act) in enumerate(COMBOS):
            ax = axs[r, c]
            q = load_ql(case, act, TF)
            if q is None:
                ax.text(0.5, 0.5, 'pending', transform=ax.transAxes, ha='center'); continue
            Jn = ql_linear_optimum(case, act, TF)
            x = q['eps'] * umax[case]; e2 = q['eps'] ** 2 * Jn
            ax.plot(x, q['J0'] / e2, LS_UNC, color=C_UNC, lw=0.9, label='uncontrolled')
            ax.plot(x, q['LQR'] / e2, LS_LQR, color=C_LQR, label='LQR')
            ax.plot(x, q['LTV'] / e2, '-', color=C_LQR, label='finite-horizon linear')
            if np.isfinite(q['LTVt']).any():
                ax.plot(x, q['LTVt'] / e2, LS_3, color=C_3, label='retuned finite-horizon linear')
            ax.plot(x, q['NLopt'] / e2, '-', color=C_OPT, lw=1.6, label='nonlinear optimum')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.axvspan(SHADE_FROM, 1e3, color='0.92', zorder=0, lw=0)
            ax.set_xlim(x.min() / 1.3, x.max() * 1.3)
            ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 3, 5]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if r == 0:
                ax.set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
            else:
                ax.set_xlabel(r'$u^\prime_{max}/u_\tau$')
        axs[r, 0].set_ylabel(rf'$J/(\varepsilon^2 J^\star_{{lin}})$, $T={TF:g}\,t_{{max}}$')
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.04), handlelength=2.0, columnspacing=1.0)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.05, y=0.12)
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    save(fig, 'fig18_ql')


if __name__ == '__main__':
    main()
