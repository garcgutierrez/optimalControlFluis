"""Figure 14 (fig:intermittent): intermittent bursts observed through the wall pressure,
T = 2 t_max, rho_hat = 1.  (a-d) fraction of the full-information cost reduction obtained
by the nonlinear optimum (solid), the certainty-equivalent controller with the Kalman
filter (dashed) and the stationary LQG (triangles); (e-h) excess cost of the Kalman-based
controller over the nonlinear optimum.

Data  : int_<case>_<actuator>_T2.pkl   (python sweep_intermittent.py <case> <actuator> 2)
Output: fig14_intermittent.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import W1, COMBOS, label, save, load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TF = 2.0                    # T/t_max
PS = [1.0, 0.1, 0.01]       # burst probabilities p shown in (a-d)
PS_GAIN = [0.3, 0.1, 0.03, 0.01]   # burst probabilities p shown in (e-h)
COLS = {1.0: '#86b6ef', 0.3: '#5598e7', 0.1: '#2a78d6', 0.03: '#1c5cab', 0.01: '#0d366b'}


def main(TF=TF):
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.5), sharex=True)
    for c, (case, act) in enumerate(COMBOS):
        O = load(f'int_{case}_{act}_T{TF:g}.pkl')
        for p in PS:
            R = sorted([r for r in O['res'] if np.isclose(r['p'], p)], key=lambda r: r['nu'])
            nu = np.array([r['nu'] for r in R]); g = lambda k: np.array([r[k] for r in R])
            J0, FI = g('unc'), g('FI_opt')
            axs[0, c].plot(nu, (J0 - g('NL')) / (J0 - FI), '-', color=COLS[p], lw=1.2, label=rf'$p={p:g}$')
            if p < 1:
                axs[0, c].plot(nu, (J0 - g('KF')) / (J0 - FI), '--', color=COLS[p], lw=1.0)
            L = [r for r in O['lqg'] if np.isclose(r['p'], p)]
            if L:
                nl = np.array([r['nu'] for r in L]); jl = np.array([r['LQG'] for r in L])
                J0p, FIp = p * O['res'][0]['unc'], p * O['res'][0]['FI_opt']
                axs[0, c].plot(nl, (J0p - jl) / (J0p - FIp), '^', color=COLS[p], ms=3.5, mfc='w', mew=0.8)
        for p in PS_GAIN:
            R = sorted([r for r in O['res'] if np.isclose(r['p'], p)], key=lambda r: r['nu'])
            nu = np.array([r['nu'] for r in R])
            gain = np.array([r['KF'] / r['NL'] - 1 for r in R]) * 100
            axs[1, c].plot(nu, gain, 'o-', color=COLS[p], ms=2.5, mfc='w', lw=1.0, label=rf'$p={p:g}$')
        axs[0, c].set_xscale('log'); axs[0, c].set_ylim(-0.05, 1.05)
        axs[1, c].set_yscale('log'); axs[1, c].set_ylim(1e-1, 300)
        axs[1, c].set_xlabel(r'$\sigma$')
        axs[0, c].set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
    axs[0, 0].set_ylabel(r'$(J^0_T-J)/(J^0_T-J^{FI}_\star)$')
    axs[1, 0].set_ylabel(r'$(J_{KF}/J_{NL}-1)\times100$')
    hs = [Line2D([], [], color=COLS[p], lw=1.3) for p in [1.0, 0.3, 0.1, 0.03, 0.01]]
    ls = [rf'$p={p:g}$' for p in [1.0, 0.3, 0.1, 0.03, 0.01]]
    hs += [Line2D([], [], color='k', lw=1.1, ls='-'), Line2D([], [], color='k', lw=1.0, ls='--'),
           Line2D([], [], color='k', ls='none', marker='^', mfc='w', ms=4)]
    ls += ['nonlinear', 'Kalman', 'stationary LQG']
    fig.legend(hs, ls, loc='upper center', ncol=8, fontsize=6.5, bbox_to_anchor=(0.5, 1.05), handlelength=1.8, columnspacing=1.0)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.84, y=0.96)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    save(fig, 'fig14_intermittent')


if __name__ == '__main__':
    main()
