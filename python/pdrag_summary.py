"""Summary of the partial-information drag study (normalised costs and advantages)."""
import common, pickle, os, numpy as np
D = common.DATA
for case in ['inner', 'outer']:
    for p in [1.0, 0.1]:
        for nu in [0.01, 0.1, 1.0]:
            f = f'{D}/pdrag_{case}_p{p:g}_nu{nu:g}.pkl'
            if not os.path.exists(f):
                continue
            d = pickle.load(open(f, 'rb'))
            fps = f'{D}/pdragps_{case}_p{p:g}_nu{nu:g}.pkl'
            ps = pickle.load(open(fps, 'rb'))['res'] if os.path.exists(fps) else []
            for x in d['res']:
                n = p * d['D0']
                nl = {k: (x['lin_exact'] + x[k][-2]) for k in x if k.startswith(('thr', 'mpc'))}
                for y in ps:
                    if np.isclose(y['gr'], x['gr']):
                        nl['ps'] = x['lin_exact'] + y['ps'][1]
                kbest = min(nl, key=nl.get)
                best = nl[kbest]
                lin = x['lin_exact']
                s = ' '.join(f'{k}={v/n:.4f}' for k, v in nl.items())
                print(f"{case} p={p:g} nu={nu:g} g/gc={x['gr']}: FI_NL={x['FI_NL']/n:.4f} FI_lin={x['FI_lin']/n:.4f} lin={lin/n:.4f} "
                      f"bestNL={best/n:.4f}({kbest}) lin/bestNL-1={100*(lin/best-1):.1f}% lin/FI_NL-1={100*(lin/x['FI_NL']-1):.1f}% "
                      f"frac(NL red)={(n-lin)/(n-best):.2f} | {s}")
