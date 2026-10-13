"""Figure 11 (fig:obsclean): control from the wall shear stress without stochastic forcing
(zeta = 0), versus the sensor noise sigma, T = 2 t_max, rho_hat = 1.  (a-d) fraction of
the full-information cost reduction obtained by each controller (dotted: full-information
LQR); (e-h) excess cost over the optimal output feedback.

Data  : obs3_<case>_<actuator>_T2.pkl   (python sweep_obs.py <case> <actuator> 2)
Output: fig11_obs_clean.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_LQR, C_OPT, C_3, C_ISO, W1, COMBOS, label, save, load_obs
import numpy as np
import matplotlib.pyplot as plt

TF = 2.0   # T/t_max


def obs_select(O, omega):
    R = [r for r in O['res'] if r['omega'] == omega]
    R.sort(key=lambda r: r['nu'])
    get = lambda k: np.array([r[k] for r in R])
    return get, np.array([r['nu'] for r in R])


def main(TF=TF):
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.4), sharex=True)
    for c, (case, act) in enumerate(COMBOS):
        O = load_obs(case, act, TF)
        get, nu = obs_select(O, 0.0)
        J0 = get('unc'); FI = get('FI_opt')
        eff = lambda J: (J0 - J) / (J0 - FI)
        ax = axs[0, c]
        ax.axhline(eff(get('FI_LQR'))[0], color=C_LQR, lw=0.7, ls=':')
        ax.plot(nu, eff(get('opt')), 'o-', color=C_OPT, ms=3, mfc='w', label='optimum')
        ax.plot(nu, eff(get('KF_LQR')), 's--', color=C_LQR, ms=3, mfc='w', label='KF + LQR')
        ax.plot(nu, eff(get('LQG_burst')), '^-.', color=C_3, ms=3, mfc='w', label='LQG, burst-shaped')
        ax.plot(nu, eff(get('LQG_iso')), 'v:', color=C_ISO, ms=3, mfc='w', label='LQG, isotropic')
        ax.set_xscale('log'); ax.set_ylim(0, 1.05)
        ax.set_title(f"{case}, {'wall' if act == 'wall' else 'distr.'}", fontsize=8)
        ax = axs[1, c]
        opt = get('opt')
        for k, col, mk, ls in [('KF_LQR', C_LQR, 's', '--'), ('LQG_burst', C_3, '^', '-.'), ('LQG_iso', C_ISO, 'v', ':')]:
            ax.plot(nu, (get(k) / opt - 1) * 100, mk + ls, color=col, ms=3, mfc='w')
        ax.set_yscale('log'); ax.set_ylim(1e-2, 300)
        ax.set_xlabel(r'$\sigma$')
    axs[0, 0].set_ylabel(r'$(J^0_T-J)/(J^0_T-J^{FI}_{\star})$'); axs[1, 0].set_ylabel(r'$(J/J_{opt}-1)\times100$')
    axs[0, 0].legend(loc='lower left', fontsize=5.8, handlelength=2.2)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.04, y=0.96)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    save(fig, 'fig11_obs_clean')


if __name__ == '__main__':
    main()
