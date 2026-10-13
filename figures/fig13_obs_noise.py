"""Figure 13 (fig:obsnoise): effect of the stochastic forcing zeta on the control from the
wall shear stress, for three levels of sensor noise, T = 2 t_max, rho_hat = 1.
(a-d) fraction of the full-information cost reduction obtained by the optimal output
feedback; (e-h) excess cost of the burst-shaped LQG (circles) and of KF + LQR (squares).

Data  : obs3_<case>_<actuator>_T2.pkl   (python sweep_obs.py <case> <actuator> 2)
Output: fig13_obs_noise.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import W1, COMBOS, label, save, load_obs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

TF = 2.0                                   # T/t_max
NUS = [1e-4, 1e-2, 1.0]                    # sensor-noise levels sigma
COLS = ['#86b6ef', '#2a78d6', '#0d366b']   # ordinal blue ramp, light -> dark = noisier sensors
OM0 = 10.0 ** -1.5                         # plotting position of zeta = 0


def main(TF=TF):
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.5), sharex=True)
    for c, (case, act) in enumerate(COMBOS):
        O = load_obs(case, act, TF)
        for k, nu in enumerate(NUS):
            R = sorted([r for r in O['res'] if np.isclose(r['nu'], nu)], key=lambda r: r['omega'])
            om = np.array([r['omega'] for r in R]); om = np.where(om == 0, OM0, om)
            g = lambda key: np.array([r[key] for r in R])
            opt, J0, FI = g('opt'), g('unc'), g('FI_opt')
            axs[0, c].plot(om, (J0 - opt) / (J0 - FI), 'o-', color=COLS[k], ms=3, mfc='w', lw=1.0, label=rf'$\sigma={nu:g}$')
            axs[1, c].plot(om, (g('LQG_burst') / opt - 1) * 100, 'o-', color=COLS[k], ms=3, mfc='w', lw=1.0)
            axs[1, c].plot(om, (g('KF_LQR') / opt - 1) * 100, 's:', color=COLS[k], ms=2.5, mfc=COLS[k], lw=0.8)
        for ax in axs[:, c]:
            ax.set_xscale('log'); ax.set_xticks([OM0, 1e0, 1e2]); ax.set_xticklabels(['0', '1', r'$10^{2}$'])
            ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs='auto', numticks=10))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.axvline(10 ** -1.25, color='k', lw=0.3)
        axs[0, c].set_ylim(0, 1.05)
        axs[1, c].set_yscale('log'); axs[1, c].set_ylim(1e-2, 300)
        axs[1, c].set_xlabel(r'$\zeta$')
        axs[0, c].set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
    axs[0, 0].set_ylabel(r'$(J^0_T-J_{opt})/(J^0_T-J^{FI}_{\star})$')
    axs[1, 0].set_ylabel(r'$(J/J_{opt}-1)\times100$')
    axs[0, 0].legend(loc='lower left', fontsize=6)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.84, y=0.96)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    save(fig, 'fig13_obs_noise')


if __name__ == '__main__':
    main()
