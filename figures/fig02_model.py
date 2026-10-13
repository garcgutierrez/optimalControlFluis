"""Figure 2 (fig:model): mean velocity profile, eddy viscosity, and maps of the maximum
transient amplification G_max and of its time t_max over (lambda_x, lambda_z).

Data  : maps_20x20.pkl           (python sweep_maps.py 20 20)
        mean profile computed on the fly (orrss.turprof_spectral)
Output: fig2_model.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, CMAP, W1, CASEMARK, label, save, load_maps, grid
import numpy as np
import matplotlib.pyplot as plt
import common
from orrss import turprof_spectral


def main():
    y, nut, uc, Ubp = turprof_spectral(512, common.RT, 1.0, 'tau')
    yp = (1 - y) * common.RT
    m = (y > 0) & (yp > 0)
    d, LX, LZ, idx = load_maps()
    G = grid(d, LX, LZ, idx, 'wall', lambda r: r['Gmax'])
    TM = grid(d, LX, LZ, idx, 'wall', lambda r: r['tmax'])
    TM[(TM <= 0) | (G < 1.05)] = np.nan
    fig = plt.figure(figsize=(W1, 4.2))
    ax1 = fig.add_axes([0.08, 0.60, 0.37, 0.36]); ax2 = fig.add_axes([0.60, 0.60, 0.37, 0.36])
    ax3 = fig.add_axes([0.08, 0.08, 0.33, 0.40]); ax4 = fig.add_axes([0.60, 0.08, 0.33, 0.40])
    cax3 = fig.add_axes([0.425, 0.08, 0.012, 0.40]); cax4 = fig.add_axes([0.945, 0.08, 0.012, 0.40])

    # (a) mean velocity profile
    ax1.semilogx(yp[m], uc[m, 0], color=C_UNC)
    yl = np.geomspace(20, 2000, 50)
    ax1.semilogx(yl, np.log(yl) / 0.41 + 5.2, ':', color=C_UNC, lw=0.8)
    ax1.set_xlabel(r'$y^+$'); ax1.set_ylabel(r'$U^+$'); ax1.set_xlim(0.5, 2000); ax1.set_ylim(0, 27)
    label(ax1, '($a$)')

    # (b) total viscosity
    ax2.loglog(yp[m], nut[m, 0] * common.RT, color=C_UNC)
    ax2.loglog(yl[:30], 0.426 * yl[:30], ':', color=C_UNC, lw=0.8)
    ax2.set_xlabel(r'$y^+$'); ax2.set_ylabel(r'$\nu_T^+$'); ax2.set_xlim(0.5, 2000); ax2.set_ylim(0.8, 300)
    label(ax2, '($b$)')

    # (c) G_max and (d) t_max maps
    for ax, cax, M, lev, lab, cl in [(ax3, cax3, G, np.r_[0.99, np.arange(1.5, 6.01, 0.5)], r'$G_{max}$', '($c$)'),
                                     (ax4, cax4, np.log10(TM), np.arange(-2.5, 0.51, 0.25), r'$\log_{10}(t_{max}u_\tau/h)$', '($d$)')]:
        Mm = np.ma.masked_invalid(M.T)
        cs = ax.contourf(LX, LZ, Mm, levels=lev, cmap=CMAP, extend='both')
        ax.contour(LX, LZ, Mm, levels=lev[::2], colors='k', linewidths=0.3, negative_linestyles='solid')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\lambda_x/h$'); ax.set_ylabel(r'$\lambda_z/h$')
        ax.plot(LX, LX, color='k', lw=0.5, ls=':')
        ax.set_xlim(LX[0], LX[-1]); ax.set_ylim(LZ[0], LZ[-1])
        for k, (mk, lx, lz) in CASEMARK.items():
            ax.plot(lx, lz, mk, mfc='w', mec='k', ms=5, mew=0.8)
        cb = fig.colorbar(cs, cax=cax)
        cb.ax.tick_params(labelsize=7); cb.set_label(lab, fontsize=8)
        label(ax, cl, y=0.97)
    save(fig, 'fig2_model')


if __name__ == '__main__':
    main()
