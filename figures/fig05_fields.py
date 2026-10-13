"""Figure 5 (fig:fields): streamwise velocity (colour) and wall-normal velocity (lines,
dashed negative) of the outer burst in the plane z = 0, T = t_max, wall transpiration,
rho_hat = 1.  Rows: uncontrolled, LQR, unrestricted optimum; columns: t = 0, T/2, T.

Data  : case_<CASE>_wall.pkl     (python cases.py outer_wall)
Output: fig5_fields_<CASE>.pdf / .png
"""
# figstyle first: it sets the BLAS threads before numpy is loaded and puts ../python on sys.path
from figstyle import W1, label, save, load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CASE = 'outer'   # 'inner' or 'outer'
TF = 1.0         # T/t_max (must be a key of R['hist'])


def main(case=CASE, TF=TF):
    R = load(f'case_{case}_wall.pkl')
    H = R['hist'][TF]
    x = H['x']; y = R['y']
    rows = [('uncontrolled', H['fu0'], H['fv0']), ('LQR', H['fuL'], H['fvL']), ('optimum', H['fuo'], H['fvo'])]
    vmax = max(np.abs(f[0][1:]).max() for f in [(H['fu0'],), (H['fuL'],), (H['fuo'],)])
    lev = np.linspace(-vmax, vmax, 17)
    RB = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#ee8f8e', '#b32d2d'])
    fig, axs = plt.subplots(3, 3, figsize=(W1, 3.4), sharex=True, sharey=True)
    times = [0, 0.5 * TF, TF]
    for i, (name, fu, fv) in enumerate(rows):
        for j in range(3):
            ax = axs[i, j]
            cs = ax.contourf(x / R['lx'], y, fu[j], levels=lev, cmap=RB, extend='both')
            vv = fv[j]; vl = np.abs(vv).max()
            ax.contour(x / R['lx'], y, vv, levels=np.linspace(-vl, vl, 8)[[1, 2, 5, 6]], colors='k', linewidths=0.4,
                       negative_linestyles='dashed')
            if i == 0:
                ax.set_title(rf'$t={times[j]:g}\,t_{{max}}$', fontsize=8)
            if j == 0:
                ax.set_ylabel(name + '\n' + r'$y/h$', fontsize=8)
            if i == 2:
                ax.set_xlabel(r'$x/\lambda_x$')
            label(ax, '($' + 'abcdefghi'[3 * i + j] + '$)', x=0.02, y=0.96)
    cax = fig.add_axes([0.93, 0.12, 0.015, 0.8])
    cb = fig.colorbar(cs, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels(['$-$max', '0', 'max']); cb.ax.tick_params(labelsize=7)
    cb.set_label(r'$u$', fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.08, hspace=0.12)
    save(fig, f'fig5_fields_{case}')


if __name__ == '__main__':
    main()
