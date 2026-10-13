"""Figure 15 (fig:drag): drag objective with an amplitude-proportional actuation cost, full
information, T = 2 t_max, rho_hat = 1.  (a-d) expected cost for Rayleigh-distributed
amplitudes, normalised with the drag increment of the uncontrolled bursts; (e-h) excess
cost of the best linear feedback over the nonlinear optimum (Rayleigh and log-normal).

Data  : drag_<case>_<actuator>_T2.pkl   (python sweep_drag.py <case> <actuator> 2)
        expected costs evaluated on the fly (drag.expected), ~1 min
Output: fig15_drag.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, C_LQR, C_OPT, C_3, W1, COMBOS, label, save, load
import numpy as np
import matplotlib.pyplot as plt
from drag import expected

TF = 2.0                          # T/t_max
GR = np.geomspace(0.02, 3, 40)    # gamma/gamma_c


def main(TF=TF):
    gr = GR
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.5), sharex=True)
    for c, (case, act) in enumerate(COMBOS):
        d = load(f'drag_{case}_{act}_T{TF:g}.pkl')
        D0, gc = d['D0'], d['gc']
        Er = np.sqrt(np.pi) / 2
        res = {dist: np.array([expected(d, g * gc, dist) for g in gr]) for dist in ['rayleigh', 'lognormal']}
        QLQ, l1LQ = d['Q'][-1], d['L1'][-1]
        ax = axs[0, c]
        ax.plot(gr, res['rayleigh'][:, 0] / D0, '-', color=C_OPT, label='nonlinear optimum')
        ax.plot(gr, res['rayleigh'][:, 1] / D0, '--', color=C_LQR, label='best linear')
        ax.plot(gr, (QLQ + gr * gc * Er * l1LQ) / D0, '-.', color=C_3, label='linear, quadratic design')
        ax.plot(gr, (d['QLQR'] + gr * gc * Er * d['l1LQR']) / D0, ':', color=C_UNC, label='energy LQR')
        ax.axhline(1, color='k', lw=0.4)
        ax.set_xscale('log'); ax.set_ylim(0.4, 1.3)
        ax.set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
        ax = axs[1, c]
        for dist, ls in [('rayleigh', '-'), ('lognormal', '--')]:
            nl, lin = res[dist][:, 0], res[dist][:, 1]
            ax.plot(gr, (lin / nl - 1) * 100, ls, color=C_LQR, lw=1.1, label=('Rayleigh' if dist == 'rayleigh' else 'log-normal'))
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-1, 30)
        ax.set_xlabel(r'$\gamma/\gamma_c$')
    axs[0, 0].set_ylabel(r'$E[J]/D^0_T$'); axs[1, 0].set_ylabel(r'$(J_{lin}/J_{NL}-1)\times100$')
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + ['excess cost, Rayleigh', 'excess cost, log-normal'], loc='upper center', ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.07), handlelength=2.2, columnspacing=1.2)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.04, y=0.96)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    save(fig, 'fig15_drag')


if __name__ == '__main__':
    main()
