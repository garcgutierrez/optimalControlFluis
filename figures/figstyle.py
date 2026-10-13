"""Shared style and helpers for the figure scripts (JFM style, vector PDF + PNG).

Every figNN_*.py script imports this module.  Edit it to change the look of *all*
figures at once (fonts, colours, text width, output folder); edit the individual
scripts to change one figure.

Output folder: ../images, or the folder given in the environment variable FIG_OUT.
"""
import os
import sys
import pickle

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # must be set before numpy is imported
os.environ.setdefault('OMP_NUM_THREADS', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', 'python'))   # computational modules (orrss, lq, obs, ...)
sys.path.insert(0, CODE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import common

D = common.DATA      # ../python/data
OUT = os.environ.get('FIG_OUT', os.path.normpath(os.path.join(HERE, '..', 'images')))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.2,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
})

# roles (reference palette, first three slots validated all-pairs) + neutral baseline
C_UNC = '#52514e'   # uncontrolled, neutral ink
C_LQR = '#2a78d6'   # slot 1 blue
C_OPT = '#eb6834'   # slot 2 orange
C_3 = '#1baf7a'     # slot 3 aqua
C_ISO = '#8a8984'   # grey (isotropic LQG)
LS_UNC, LS_LQR, LS_OPT, LS_3 = '-', '--', '-', '-.'
BLUES = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('seqblue', BLUES)
RAMP5 = ['#b7d3f6', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b']   # ordinal blue ramp
RH_COL = {0: C_LQR, 1: C_OPT, 2: C_3}                               # the three control penalties rho_hat
W1 = 5.3   # full text width (in)

# representative cases: marker, lambda_x/h, lambda_z/h
CASEMARK = {'inner': ('o', 0.5, 0.05), 'outer': ('s', 10.0, 3.0)}
COMBOS = [('inner', 'wall'), ('inner', 'distributed'), ('outer', 'wall'), ('outer', 'distributed')]


def label(ax, s, x=0.03, y=0.95, **kw):
    """Panel label, e.g. label(ax, '($a$)')."""
    ax.text(x, y, s, transform=ax.transAxes, va='top', ha='left', fontsize=9, **kw)


def save(fig, name):
    """Write OUT/name.pdf and OUT/name.png."""
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
    plt.close(fig)
    print('wrote', os.path.join(OUT, name + '.pdf'))


def load(fname):
    """Load a pickle from the data folder."""
    with open(os.path.join(D, fname), 'rb') as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# wavenumber maps (figures 2, 6, 7)
# ----------------------------------------------------------------------------
def load_maps(fname='maps_20x20.pkl'):
    d = load(fname)
    LX, LZ = d['LX'], d['LZ']
    idx = {(r['act'], round(r['lx'], 10), round(r['lz'], 10)): r for r in d['res']}
    return d, LX, LZ, idx


def grid(d, LX, LZ, idx, act, f):
    """Array M[i, j] = f(result at LX[i], LZ[j]) for actuator act (NaN where missing)."""
    M = np.full((len(LX), len(LZ)), np.nan)
    for i, lx in enumerate(LX):
        for j, lz in enumerate(LZ):
            r = idx.get((act, round(lx, 10), round(lz, 10)))
            if r is None or 'err' in r:
                continue
            try:
                M[i, j] = f(r)
            except Exception:
                pass
    return M


# ----------------------------------------------------------------------------
# output feedback (figures 11, 13)
# ----------------------------------------------------------------------------
def load_obs(case, act, TF):
    fn = f'obs3_{case}_{act}_T{TF:g}.pkl'
    if not os.path.exists(os.path.join(D, fn)):
        fn = f'obs_{case}_{act}_T{TF:g}.pkl'
    return load(fn)
