"""Summary of the quasi-linear study."""
import sys, os
sys.argv = [sys.argv[0]]
import common
import numpy as np, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
from fig18_ql import load_ql, ql_linear_optimum
from figstyle import COMBOS, D
umax = {c: float(pickle.load(open(os.path.join(D, f'qldiag_{c}.pkl'), 'rb'))['urms'].max()) for c in ['inner', 'outer']}
for case, act in COMBOS:
    for TF in [1.0, 2.0]:
        q = load_ql(case, act, TF)
        if q is None:
            continue
        Jn = ql_linear_optimum(case, act, TF)
        print(f'--- {case} {act} T={TF:g} tmax')
        for i, e in enumerate(q['eps']):
            n = q['NLopt'][i]
            print(f"  eps={e:<5g} u'={e*umax[case]:5.2f}: NL/lin={n/(e**2*Jn):.3f} J0/NL={q['J0'][i]/n:.3f} LQR/NL={q['LQR'][i]/n:.4f} "
                  f"LTV/NL={q['LTV'][i]/n:.4f} open/NL={q['open'][i]/n:.4f} LQRt/NL={q['LQRt'][i]/n:.4f} LTVt/NL={q['LTVt'][i]/n:.4f}")
