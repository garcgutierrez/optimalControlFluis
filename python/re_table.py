"""LaTeX rows of the Reynolds-number table (appendix) from data/re_*.pkl: one row per Reynolds number,
the range of G_max over the resolutions tested, and the ratios at the finest resolution available."""
import common, pickle, glob, os, re
import numpy as np
D = common.DATA
def runs(case, act, Rt):
    out = {}
    for f in glob.glob(f'{D}/re_{case}_{act}_{Rt}_N*.pkl'):
        N = int(re.search(r'_N(\d+)\.pkl', f).group(1)); out[N] = pickle.load(open(f, 'rb'))
    return out
for case in ['inner', 'outer']:
    for Rt in [550, 1000, 2000, 5000, 10000]:
        w, d = runs(case, 'wall', Rt), runs(case, 'distributed', Rt)
        Ns = sorted(set(w) | set(d))
        G = [x['Gmax'] for x in list(w.values()) + list(d.values())]
        rw, rd = w[max(w)], d[max(d)]
        tm = f"{rw['tmax']*Rt:.1f}" if case == 'inner' else f"{rw['tmax']:.3f}"
        Gs = f"{max(G):.3f}" if max(G) - min(G) < 0.002 else f"{min(G):.3f}--{max(G):.3f}"
        Nsr = f"{Ns[0]}" if len(Ns) == 1 else f"{Ns[0]}--{Ns[-1]}"
        Rs = {550: '550', 1000: '1000', 2000: '2000', 5000: '5000', 10000: '$10^4$'}[Rt]
        f3 = lambda x: f'{x:.3f}'
        print(f" & {Rs} & {Nsr} & {Gs} & {tm} & {f3(rw['Gamma'][1])} & {f3(rw['Gamma'][2])} & {f3(rw['Disc'][0])} & "
              f"{f3(rd['Gamma'][1])} & {f3(rd['Gamma'][2])} & {f3(rd['Disc'][0])} \\\\")
