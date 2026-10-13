"""Figure 8 (fig:random): loss of the LQR for random, exponentially distributed lifetimes
with mean tau_L (solid: inner; dashed: outer), compared with a deterministic lifetime
T = tau_L (dotted, inner).  (a) wall transpiration; (b) distributed forcing.

Data  : case_{inner,outer}_{wall,distributed}.pkl
Output: fig8_random.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import D, RH_COL, W1, label, save, load
import os
import numpy as np
import matplotlib.pyplot as plt

RHATS = {'wall': [0.01, 1.0, 100.0], 'distributed': [0.1, 1.0, 10.0]}


def main():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.2), sharey=True)
    for c, act in enumerate(['wall', 'distributed']):
        for ir, rh in enumerate(RHATS[act]):
            col = RH_COL[ir]
            for case, ls in [('inner', '-'), ('outer', '--')]:
                fn = f'case_{case}_{act}.pkl'
                if not os.path.exists(os.path.join(D, fn)):
                    continue
                R = load(fn)
                k = np.argmin(abs(np.log(R['rhats'] / rh)))
                Dd = R['disc'][k]; S = R['scan'][k]
                axs[c].plot(R['TFs'], (Dd[:, 1] / Dd[:, 2] - 1) * 100, ls, color=col, lw=1.1,
                            label=(rf'$\hat\rho={rh:g}$' if case == 'inner' else None))
                if case == 'inner':
                    axs[c].plot(R['TFs'], (S[:, 1] / S[:, 2] - 1) * 100, ':', color=col, lw=0.8)
        axs[c].set_xscale('log'); axs[c].set_yscale('log'); axs[c].set_xlim(0.1, 10); axs[c].set_ylim(1e-2, 300)
        axs[c].set_xlabel(r'$\tau_L/t_{max}$,  $T/t_{max}$')
        axs[c].set_title('wall transpiration' if act == 'wall' else 'distributed forcing', fontsize=8.5)
        axs[c].legend(loc='lower left', fontsize=7)
        label(axs[c], '($' + 'ab'[c] + '$)', x=0.88, y=0.95)
    axs[0].set_ylabel(r'$(\Gamma-1)\times100$')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig8_random')


if __name__ == '__main__':
    main()
