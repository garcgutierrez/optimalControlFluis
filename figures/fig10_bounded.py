"""Figure 10 (fig:bounded): cost of bounded actuation, normalised with the unconstrained
optimum, versus the bound phi_max (normalised with the largest amplitude of the
unconstrained LQR control).  Orange: constrained optimum; blue: saturated LQR; green:
detuned LQR.  Circles T = t_max, squares T = 3 t_max; dotted: uncontrolled.

Data  : case_inner_wall.pkl, case_outer_wall.pkl, case_outer_distributed.pkl
Output: fig10_bounded.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import D, C_UNC, C_LQR, C_OPT, C_3, W1, label, save, load
import os
import matplotlib.pyplot as plt

PANELS = [('inner', 'wall'), ('outer', 'wall'), ('outer', 'distributed')]
LIFETIMES = [(1.0, 'o'), (3.0, 's')]     # (T/t_max, marker); must be keys of R['bounded']


def main():
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.0))
    k = 0
    for case, act in PANELS:
        fn = f'case_{case}_{act}.pkl'
        if not os.path.exists(os.path.join(D, fn)):
            k += 1; continue
        R = load(fn)
        ax = axs[k]
        for TF, mk in LIFETIMES:
            B = R['bounded'][TF]
            ref = B['Jd']
            ax.plot(B['fr'], B['Jc'] / ref, mk + '-', color=C_OPT, ms=3.5, mfc='w', lw=1.0,
                    label=('constrained optimum' if TF == 3.0 else None))
            ax.plot(B['fr'], B['Js'] / ref, mk + '--', color=C_LQR, ms=3.5, mfc='w', lw=1.0,
                    label=('saturated LQR' if TF == 3.0 else None))
            ax.plot(B['fr'], B['Jdet'] / ref, mk + '-.', color=C_3, ms=3.5, mfc='w', lw=1.0,
                    label=('detuned LQR' if TF == 3.0 else None))
            ax.axhline(B['J0'] / ref, color=C_UNC, lw=0.6, ls=':')
        ax.set_xscale('log'); ax.set_xlabel(r'$\phi_{max}/\phi_{ref}$')
        ax.set_title(f'{case}, ' + ('wall' if act == 'wall' else 'distributed'), fontsize=8)
        label(ax, '($' + 'abc'[k] + '$)', x=0.85, y=0.95)
        k += 1
    axs[0].set_ylabel(r'$J_T/J^\star_T$')
    axs[0].legend(loc='upper right', fontsize=6.5, bbox_to_anchor=(1.0, 0.85))
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig10_bounded')


if __name__ == '__main__':
    main()
