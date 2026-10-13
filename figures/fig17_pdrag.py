"""Figure 17 (fig:pdrag): drag objective with an amplitude-proportional actuation cost,
controlled from noisy wall-pressure measurements, wall transpiration, T = 2 t_max,
rho_hat = 1.  Expected cost normalised with the drag increment of the uncontrolled
bursts, versus the sensor noise sigma; rows: burst probability p = 1, 0.1.  The shaded
band spans the full-information nonlinear optimum and the best nonlinear output
feedback found (thresholded certainty-equivalent, MPC or direct policy search).

Data  : pdrag_<case>_p<p>_nu<nu>.pkl     (python sweep_pdrag.py <case> <p> <nu>)
        pdragps_<case>_p<p>_nu<nu>.pkl   (python pdrag_extra.py <case> <p> <nu>, optional)
Output: fig17_pdrag.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import D, C_UNC, C_LQR, C_OPT, W1, label, save, load
import os
import numpy as np
import matplotlib.pyplot as plt

NUS = [0.01, 0.1, 1.0]                                                    # sensor noise sigma
PANELS = [('inner', 0.3), ('inner', 0.7), ('outer', 0.3), ('outer', 0.7)]  # (case, gamma/gamma_c)
PROBS = [1.0, 0.1]                                                        # burst probability p (rows)


def load_pdrag(case, p, nu):
    f = f'pdrag_{case}_p{p:g}_nu{nu:g}.pkl'
    if not os.path.exists(os.path.join(D, f)):
        return None
    d = load(f)
    fps = f'pdragps_{case}_p{p:g}_nu{nu:g}.pkl'
    ps = load(fps)['res'] if os.path.exists(os.path.join(D, fps)) else None
    return d, ps


def main():
    fig, axs = plt.subplots(2, 4, figsize=(W1, 3.6), sharex=True)
    for r, p in enumerate(PROBS):
        for c, (case, gr) in enumerate(PANELS):
            ax = axs[r, c]
            rows = []
            for nu in NUS:
                L = load_pdrag(case, p, nu)
                if L is None:
                    continue
                d, ps = L
                x = [z for z in d['res'] if np.isclose(z['gr'], gr)][0]
                n = p * d['D0']
                nl = {k: x['lin_exact'] + x[k][-2] for k in x if k.startswith(('thr', 'mpc'))}
                if ps is not None:
                    y = [z for z in ps if np.isclose(z['gr'], gr)]
                    if y:
                        nl['ps'] = x['lin_exact'] + y[0]['ps'][1]
                rows.append((nu, x['FI_NL'] / n, x['FI_lin'] / n, x['lin_exact'] / n, min(nl.values()) / n))
            if not rows:
                ax.text(0.5, 0.5, 'pending', transform=ax.transAxes, ha='center'); continue
            R = np.array(rows)
            ax.fill_between(R[:, 0], R[:, 1], R[:, 4], color=C_OPT, alpha=0.15, lw=0)
            ax.plot(R[:, 0], R[:, 2], ':', color=C_LQR, lw=1.0, label='best linear, full information')
            ax.plot(R[:, 0], R[:, 1], ':', color=C_OPT, lw=1.0, label='nonlinear optimum, full information')
            ax.plot(R[:, 0], R[:, 3], 'o--', color=C_LQR, ms=3.5, mfc='w', label='best linear output feedback')
            ax.plot(R[:, 0], R[:, 4], 's-', color=C_OPT, ms=3.5, label='best nonlinear output feedback found')
            ax.axhline(1.0, color=C_UNC, lw=0.6)
            ax.set_xscale('log')
            if r == 0:
                ax.set_title(rf"{case}, $\gamma={gr:g}\gamma_c$", fontsize=8)
            else:
                ax.set_xlabel(r'$\sigma$')
        axs[r, 0].set_ylabel(rf'$J/(pD^0_T)$, $p={p:g}$')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.07), handlelength=2.2, columnspacing=1.2)
    for i, ax in enumerate(axs.ravel()):
        label(ax, '($' + 'abcdefgh'[i] + '$)', x=0.05, y=0.95)
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    save(fig, 'fig17_pdrag')


if __name__ == '__main__':
    main()
