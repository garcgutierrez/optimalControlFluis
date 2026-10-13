"""Figure 3 (fig:verification): verification for the outer case with wall transpiration,
rho_hat = 1, T = t_max.  (a) L-BFGS convergence; (b) time-step convergence of the
costs; (c) control history: LQR, closed-form optimum and direct optimisation.

Data  : case_outer_wall.pkl      (python cases.py outer_wall)
Output: fig3_verification.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_LQR, C_OPT, C_3, LS_LQR, LS_OPT, W1, label, save, load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def main():
    R = load('case_outer_wall.pkl')
    v = R['ver']
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.85))

    # (a) convergence of the optimiser
    ax = axs[0]
    h = v['hist']
    ax.semilogy(np.arange(1, len(h) + 1), (h - v['Jd'][v['Nt'].index(200)]) / v['Jo'], color=C_OPT)
    ax.set_xlabel('function evaluations'); ax.set_ylabel(r'$(J^{(k)}-J^\star_h)/J^\star$')
    ax.set_xlim(0, len(h))
    label(ax, '($a$)', x=0.75)

    # (b) time-step convergence
    ax = axs[1]
    Nt = np.array(v['Nt']); T = v['T']
    err = np.abs(np.array(v['Jdo']) - v['Jo']) / v['Jo']
    errL = np.abs(np.array(v['JLd']) - v['JL']) / v['JL']
    ax.loglog(T / Nt / R['tmax'], err, 'o-', color=C_OPT, ms=4, mfc='w', label='optimum')
    ok = errL < 1
    ax.loglog((T / Nt / R['tmax'])[ok], errL[ok], 's--', color=C_LQR, ms=4, mfc='w', label='LQR')
    hh = T / Nt / R['tmax']
    ax.loglog(hh, 0.5 * err[-1] * (hh / hh[-1]) ** 2, ':', color='k', lw=0.7)
    ax.text(hh[3], 0.5 * err[-1] * (hh[3] / hh[-1]) ** 2 * 0.15, r'$\propto\Delta t^2$', fontsize=7)
    ax.loglog(hh, 2 * errL[-1] * (hh / hh[-1]), ':', color='k', lw=0.7)
    ax.text(hh[-1] * 1.0, 2 * errL[-1] * 2.2, r'$\propto\Delta t$', fontsize=7)
    ax.set_xlabel(r'$\Delta t/t_{max}$'); ax.set_ylabel('relative error in $J$')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, -0.02), fontsize=6.5, handlelength=2.2)
    label(ax, '($b$)', x=0.05, y=0.95)

    # (c) control histories
    ax = axs[2]
    H = R['hist'][1.0]
    ax.plot(H['t'] / R['tmax'], H['UL'][:, 0], LS_LQR, color=C_LQR, label='LQR')
    ax.plot(H['tt'] / R['tmax'], H['Uo'][:, 0], LS_OPT, color=C_OPT, label='optimum')
    ax.plot(v['t_d'][::6] / R['tmax'], v['U_direct'][::6, 0], 'o', ms=3, mfc='none', mec=C_3, mew=0.7, label='direct (L-BFGS)')
    ax.set_xlabel(r'$t/t_{max}$'); ax.set_ylabel(r'$|\hat{g}_+|/u_\tau$')
    ax.set_yscale('log'); ax.set_xlim(0, 1.5); ax.set_ylim(1e-3, 2)
    ax.legend(loc='lower left', fontsize=6.5)
    label(ax, '($c$)', x=0.85, y=0.95)

    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig3_verification')


if __name__ == '__main__':
    main()
