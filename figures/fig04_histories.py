"""Figure 4 (fig:histories): energy (a-d) and wall-velocity amplitude (e-h) of the bursts
controlled by wall transpiration, rho_hat = 1, for the inner and outer cases and
T = t_max, 2 t_max.  Uncontrolled, LQR and unrestricted optimum.

Data  : case_inner_wall.pkl, case_outer_wall.pkl   (python cases.py inner_wall outer_wall)
Output: fig4_histories.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import C_UNC, C_LQR, C_OPT, LS_UNC, LS_LQR, LS_OPT, W1, label, save, load
import matplotlib.pyplot as plt

CASES = ['inner', 'outer']
LIFETIMES = [1.0, 2.0]     # T/t_max (must be keys of R['hist'])


def main():
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.2), sharex='col')
    for c, case in enumerate(CASES):
        R = load(f'case_{case}_wall.pkl')
        tm = R['tmax']
        for k, TF in enumerate(LIFETIMES):
            H = R['hist'][TF]
            # energy
            ax = axs[0, 2 * c + k]
            ax.plot(H['t'] / tm, H['E0'], LS_UNC, color=C_UNC, lw=0.9, label='uncontrolled')
            ax.plot(H['t'] / tm, H['EL'], LS_LQR, color=C_LQR, label='LQR')
            ax.plot(H['tt'] / tm, H['Eo'], LS_OPT, color=C_OPT, label='optimum')
            ax.axvline(TF, color='k', lw=0.4)
            ax.set_ylim(0, None)
            ax.set_title(case + rf', $T={TF:g}\,t_{{max}}$', fontsize=8)
            # control amplitude
            ax = axs[1, 2 * c + k]
            ax.plot(H['t'] / tm, H['UL'][:, 0], LS_LQR, color=C_LQR)
            ax.plot(H['tt'] / tm, H['Uo'][:, 0], LS_OPT, color=C_OPT)
            ax.axvline(TF, color='k', lw=0.4)
            ax.set_xlabel(r'$t/t_{max}$')
            ax.set_yscale('log')
            m0 = H['UL'][:, 0].max()
            ax.set_ylim(3e-3 * m0, 2 * m0)
    axs[0, 0].set_ylabel(r'$E/E_0$'); axs[1, 0].set_ylabel(r'$|\hat g_\pm|/u_\tau$')
    axs[0, 0].legend(loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=6, handlelength=2.0)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.04 if i < 4 else 0.80, y=0.96)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    save(fig, 'fig4_histories')


if __name__ == '__main__':
    main()
