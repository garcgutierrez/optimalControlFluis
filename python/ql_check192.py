"""Resolution check of the quasi-linear study: full comparison at N=192 for a few amplitudes."""
import sys, os, shutil
import common
import ql_study
case, act, TF = sys.argv[1], sys.argv[2], float(sys.argv[3])
eps = [float(e) for e in sys.argv[4].split(',')]
ql_study.run(case, act, TF, eps, '_tmpN192', N=192)
src = f'{common.DATA}/ql_{case}_{act}_T{TF:g}_tmpN192.pkl'
shutil.move(src, f'{common.DATA}/qlcheckN192_{case}_{act}_T{TF:g}.pkl')
