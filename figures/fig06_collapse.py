"""Figure 6 (fig:collapse): (a,b) loss of the LQR, Gamma - 1, and (c,d) fraction of the
achievable cost reduction obtained by the LQR, versus T/t_max.  Bands: 5-95 percentiles
over all wavenumbers of the maps with G_max >= 1.5 (circles: median); lines: the inner
(solid) and outer (dashed) representative cases.  (a,c) wall transpiration; (b,d)
distributed forcing.

Data  : maps_20x20.pkl, case_{inner,outer}_{wall,distributed}.pkl
Output: fig6_collapse.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import D, RH_COL, W1, label, save, load, load_maps
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    d, LX, LZ, idx = load_maps()
    TF = d['TF']
    fig, axs = plt.subplots(2, 2, figsize=(W1, 3.9), sharex=True)
    for c, act in enumerate(['wall', 'distributed']):
        rhs = d['RHAT'][act]
        for ir, rh in enumerate(rhs):
            G_all = []; E_all = []
            for r in d['res']:
                if r['act'] != act or 'err' in r or not r['res'] or r['Gmax'] < 1.5:
                    continue
                q = r['res'][ir]
                if 'err' in q:
                    continue
                G_all.append(q['JL'] / q['Jo']); E_all.append((q['J0'] - q['JL']) / (q['J0'] - q['Jo']))
            G_all = np.array(G_all); E_all = np.array(E_all)
            # spread over wavenumbers: 5-95 percentiles
            lo, md, hi = np.percentile(G_all - 1, [5, 50, 95], axis=0)
            col = RH_COL[ir]
            axs[0, c].fill_between(TF, np.maximum(lo, 1e-6) * 100, hi * 100, color=col, alpha=0.18, lw=0)
            axs[0, c].plot(TF, md * 100, 'o', color=col, ms=3.5, mfc='w', mew=0.8)
            lo, md, hi = np.percentile(E_all, [5, 50, 95], axis=0)
            axs[1, c].fill_between(TF, lo, hi, color=col, alpha=0.18, lw=0)
            axs[1, c].plot(TF, md, 'o', color=col, ms=3.5, mfc='w', mew=0.8)
            for case, ls in [('inner', '-'), ('outer', '--')]:
                fn = f'case_{case}_{act}.pkl'
                if not os.path.exists(os.path.join(D, fn)):
                    continue
                R = load(fn)
                k = np.argmin(abs(np.log(R['rhats'] / rh)))
                S = R['scan'][k]
                axs[0, c].plot(R['TFs'], (S[:, 1] / S[:, 2] - 1) * 100, ls, color=col, lw=1.0,
                               label=(rf'$\hat\rho={rh:g}$' if case == 'inner' else None))
                axs[1, c].plot(R['TFs'], (S[:, 0] - S[:, 1]) / (S[:, 0] - S[:, 2]), ls, color=col, lw=1.0)
        axs[0, c].set_yscale('log'); axs[0, c].set_ylim(1e-2, 300)
        axs[0, c].set_title('wall transpiration' if act == 'wall' else 'distributed forcing', fontsize=8.5)
        axs[1, c].set_ylim(-1.0, 1.05); axs[1, c].axhline(0, color='k', lw=0.4); axs[1, c].axhline(1, color='k', lw=0.4, ls=':')
        axs[1, c].set_xlabel(r'$T/t_{max}$'); axs[1, c].set_xscale('log'); axs[1, c].set_xlim(0.1, 10)
        axs[0, c].legend(loc='lower left', fontsize=7)
    axs[0, 0].set_ylabel(r'$(\Gamma-1)\times100$'); axs[1, 0].set_ylabel(r'$\mathcal{E}$')
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcd'[i] + '$)', x=0.88, y=0.95)
    fig.tight_layout(w_pad=0.8, h_pad=0.5)
    save(fig, 'fig6_collapse')


if __name__ == '__main__':
    main()
