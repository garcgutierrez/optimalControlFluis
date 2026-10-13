"""Figure 16 (fig:dragcontrols): amplitude of the optimal wall velocity at the upper wall for
the drag objective with an amplitude-proportional cost, T = 2 t_max, for three values of
the effective weight gamma_tilde/gamma_c, compared with the energy LQR.

Data  : drag_{inner,outer}_wall_T2.pkl   (python sweep_drag.py <case> wall 2)
Output: fig16_drag_controls.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, C_LQR, C_OPT, C_3, W1, label, save, load
import numpy as np
import matplotlib.pyplot as plt

TF = 2.0                     # T/t_max
GFRAC = (0.03, 0.3)          # gamma_tilde/gamma_c shown besides 0 (nearest available value is used)


def main(TF=TF):
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.9), sharey=False)
    for k, case in enumerate(['inner', 'outer']):
        d = load(f'drag_{case}_wall_T{TF:g}.pkl')
        t = d['t'] / d['tmax']; gc = d['gc']
        allk = np.array(sorted(d['Us'].keys()))
        keys = [0.0] + [allk[np.argmin(np.abs(allk - f * gc))] for f in GFRAC]
        styles = {0: ('-', C_LQR), 1: ('--', C_3), 2: ('-', C_OPT)}
        for j, g in enumerate(keys):
            U = d['Us'][g]
            ls, col = styles.get(j, ('-', 'k'))
            axs[k].plot(t, np.abs(U[:, 0]), ls, color=col, lw=1.1, label=rf'$\tilde\gamma/\gamma_c={g/gc:.2g}$')
        axs[k].plot(t, np.abs(d['ULQR'][:, 0]), '-', color=C_UNC, lw=0.7, label='energy LQR')
        axs[k].set_yscale('log'); axs[k].set_xlabel(r'$t/t_{max}$')
        m0 = max(np.abs(d['Us'][keys[0]][:, 0]).max(), 1e-12)
        axs[k].set_ylim(1e-3 * m0, 3 * m0)
        axs[k].set_title(f'{case}, wall', fontsize=8)
        label(axs[k], '($' + 'ab'[k] + '$)', x=0.88, y=0.95)
    axs[0].set_ylabel(r'$|\hat g_+|/u_\tau$')
    axs[1].legend(loc='lower left', fontsize=6)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig16_drag_controls')


if __name__ == '__main__':
    main()
