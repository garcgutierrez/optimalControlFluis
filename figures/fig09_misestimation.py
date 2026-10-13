"""Figure 9 (fig:mis): excess cost of a controller designed for the wrong lifetime, with
respect to the optimum for the true one, rho_hat = 1.  (a-d) deterministic lifetime
(design T_d, actual T); (e-h) random lifetime (design tau_d, actual tau_L).  The point
at 'infinity' is the LQR.

Data  : mis_{inner,outer}_{wall,distributed}.pkl   (python misestimation.py <case> <actuator>)
Output: fig9_misestimation.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import RAMP5, W1, COMBOS, label, save, load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator

XINF = 45.0                       # abscissa where the infinite design lifetime (LQR) is plotted
ACT = [0.5, 1.0, 2.0, 5.0]        # actual lifetimes T/t_max (or tau_L/t_max) shown


def main():
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.5), sharex=True, sharey=True)
    cols = [RAMP5[0], RAMP5[1], RAMP5[3], RAMP5[4]]
    for c, (case, act) in enumerate(COMBOS):
        d = load(f'mis_{case}_{act}.pkl')
        for row, (Rm, Jo, TX, TDx, mk) in enumerate([(d['R'], d['Jopt'], d['TA'], d['TD'], 'o'),
                                                    (d['Rr'], d['Jr'], d['TL'], d['TDR'], 's')]):
            ax = axs[row, c]
            E = np.maximum(Rm / Jo[None, :] - 1, 0) * 100
            fin = np.isfinite(TDx)
            for k, ta in enumerate(ACT):
                j = int(np.argmin(np.abs(TX - ta)))
                x = np.r_[TDx[fin] / ta, XINF]
                ax.plot(x[:-1], E[fin, j], mk + '-', color=cols[k], ms=3, mfc='w', lw=1.0,
                        label=rf'${ta:g}\,t_{{max}}$')
                ax.plot([x[-2], XINF], [E[fin, j][-1], E[~fin, j][0]], ':', color=cols[k], lw=0.8)
                ax.plot(XINF, E[~fin, j][0], mk, color=cols[k], ms=4)
            ax.axvline(1.0, color='0.6', lw=0.6, zorder=0)
            ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=0.1, linscale=0.5)
            ax.set_xlim(0.04, 70); ax.set_ylim(0, 300)
            ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, XINF]))
            ax.set_xticklabels(['$10^{-1}$', '$1$', '$10$', r'$\infty$'])
            ax.xaxis.set_minor_formatter(NullFormatter())
        axs[0, c].set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
        axs[1, c].set_xlabel(r'$T_d/T$  or  $\tau_d/\tau_L$')
    axs[0, 0].set_ylabel('fixed lifetime\n' + r'$J/J^\star-1$ (%)', labelpad=2)
    axs[1, 0].set_ylabel('random lifetime\n' + r'$\langle J\rangle/\langle J\rangle^\star-1$ (%)', labelpad=2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, [r'$T,\ \tau_L=$' + l[0]] + l[1:], loc='upper center', ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.04), handlelength=1.8, columnspacing=1.2)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.05, y=0.1)
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    save(fig, 'fig9_misestimation')


if __name__ == '__main__':
    main()
