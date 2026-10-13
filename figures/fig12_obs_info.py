"""Figure 12 (fig:obsinfo): (a) energy of the uncontrolled bursts (grey) and error variance
of the optimal filter, tr Sigma = E_unc/(1 + chi), for two levels of sensor noise;
(b) accumulated signal-to-noise ratio of the wall shear stress.

Data  : none -- computed on the fly from the linear model (lq.Plant, obs.wall_shear_C), ~10 s
Output: fig12_obs_info.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, W1, CASEMARK, label, save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid
import common
from lq import Plant
from obs import wall_shear_C

NOISE = [(1.0, '#86b6ef'), (1e-2, '#1c5cab')]   # (sensor noise sigma, colour)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.9))
    for case, ls in [('inner', '-'), ('outer', '--')]:
        lx, lz = CASEMARK[case][1], CASEMARK[case][2]
        pl = Plant(2 * np.pi / lx, 2 * np.pi / lz, N=common.NCHEB, actuator='wall', profile=common.profile())
        tmax, Gmax, z0, _, _ = pl.transient_growth()
        C = wall_shear_C(pl)
        t = np.linspace(0, 3 * tmax, 301)
        w, V = np.linalg.eig(pl.AS)
        xi0 = pl.Zphys.conj().T @ z0
        c = np.linalg.solve(V, xi0)
        Z = np.array([pl.Zphys @ (V @ (np.exp(w * tk) * c)) for tk in t])
        E = np.sum(np.abs(Z) ** 2, 1)
        sig = np.sum(np.abs(Z @ C.T) ** 2, 1)
        cum = cumulative_trapezoid(sig, t, initial=0)
        cum /= np.interp(tmax, t, cum)
        axs[0].plot(t / tmax, E, ls, color=C_UNC)
        axs[1].plot(t / tmax, cum, ls, color=C_UNC, label=case)
        for nu, col in NOISE:
            h = 4 / nu * cum
            axs[0].plot(t / tmax, E / (1 + h), ls, color=col, lw=0.9)
    hs = [Line2D([], [], color=C_UNC, lw=1.2), Line2D([], [], color=NOISE[0][1], lw=0.9), Line2D([], [], color=NOISE[1][1], lw=0.9),
          Line2D([], [], color='k', ls='-', lw=0.8), Line2D([], [], color='k', ls='--', lw=0.8)]
    ls_ = [r'$E_{unc}$', rf'$\mathrm{{tr}}\,\Sigma$, $\sigma={NOISE[0][0]:g}$', rf'$\mathrm{{tr}}\,\Sigma$, $\sigma={NOISE[1][0]:g}$', 'inner', 'outer']
    axs[0].set_xlabel(r'$t/t_{max}$'); axs[0].set_ylabel(r'$E_{unc}$,  $\mathrm{tr}\,\Sigma$')
    axs[1].set_xlabel(r'$t/t_{max}$'); axs[1].set_ylabel(r'$\chi\,\sigma/4$')
    axs[0].set_yscale('log'); axs[0].set_ylim(1e-4, 5)
    axs[0].legend(hs, ls_, loc='lower left', ncol=2, fontsize=6, handlelength=1.8, columnspacing=1.0)
    axs[1].legend(loc='lower right', fontsize=7); axs[1].axhline(1, color='k', lw=0.4, ls=':')
    label(axs[0], '($a$)', x=0.88, y=0.95); label(axs[1], '($b$)', x=0.05, y=0.95)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig12_obs_info')


if __name__ == '__main__':
    main()
