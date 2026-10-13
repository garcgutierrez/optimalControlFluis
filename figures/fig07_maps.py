"""Figure 7 (fig:maps): loss of the LQR, (Gamma - 1) x 100, over (lambda_x, lambda_z) for
rho_hat = 1.  (a,b) wall transpiration; (c,d) distributed forcing; (a,c) T = t_max;
(b,d) T = 2 t_max.  Dotted: isolines of G_max = 1.5, 2.5, 3.5, 4.5.

Data  : maps_20x20.pkl           (python sweep_maps.py 20 20)
Output: fig7_maps.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, CMAP, W1, CASEMARK, label, save, load_maps, grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

LEVELS = [0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75]   # contour levels of (Gamma - 1) x 100
IRHO = 1                                                      # index of rho_hat in d['RHAT'][act] (1 -> rho_hat = 1)


def main():
    d, LX, LZ, idx = load_maps()
    TF = list(d['TF'])
    fig = plt.figure(figsize=(W1, 4.1))
    lev = LEVELS
    cmap = CMAP
    norm = BoundaryNorm(lev, cmap.N, extend='max')
    k = 0
    for r_, (act, ir) in enumerate([('wall', IRHO), ('distributed', IRHO)]):
        for c_, tf in enumerate([1.0, 2.0]):
            ax = fig.add_axes([0.08 + c_ * 0.425, 0.55 - r_ * 0.47, 0.37, 0.37])
            it = TF.index(tf)
            M = grid(d, LX, LZ, idx, act, lambda r: (r['res'][ir]['JL'][it] / r['res'][ir]['Jo'][it] - 1) * 100)
            Gm = grid(d, LX, LZ, idx, act, lambda r: r['Gmax'])
            M[Gm < 1.05] = np.nan
            cs = ax.contourf(LX, LZ, np.ma.masked_invalid(M.T), levels=lev, cmap=cmap, norm=norm, extend='max')
            ax.contour(LX, LZ, np.ma.masked_invalid(M.T), levels=lev, colors='k', linewidths=0.25)
            ax.contour(LX, LZ, Gm.T, levels=[1.5, 2.5, 3.5, 4.5], colors=C_UNC, linewidths=0.6, linestyles=':')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(LX[0], LX[-1]); ax.set_ylim(LZ[0], LZ[-1])
            ax.plot(LX, LX, color='k', lw=0.5, ls=':')
            for kk, (mk, lx, lz) in CASEMARK.items():
                ax.plot(lx, lz, mk, mfc='w', mec='k', ms=4.5, mew=0.8)
            if r_ == 1:
                ax.set_xlabel(r'$\lambda_x/h$')
            else:
                ax.set_xticklabels([])
            if c_ == 0:
                ax.set_ylabel(r'$\lambda_z/h$')
            else:
                ax.set_yticklabels([])
            ax.set_title(('wall transpiration' if act == 'wall' else 'distributed forcing') + rf', $T={tf:g}\,t_{{max}}$', fontsize=8, pad=3)
            label(ax, '($' + 'abcd'[k] + '$)', x=0.03, y=0.97)
            k += 1
    cax = fig.add_axes([0.90, 0.08, 0.015, 0.84])
    cb = fig.colorbar(cs, cax=cax, ticks=lev)
    cb.ax.set_yticklabels([f'{v:g}' for v in lev])
    cb.set_label(r'$(\Gamma-1)\times 100$', fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, 'fig7_maps')


if __name__ == '__main__':
    main()
