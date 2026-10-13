"""Validate the Python port against the MATLAB results in New_res2e3_0.mat."""
import numpy as np, scipy.io, scipy.linalg as sla, time
from orrss import OrrModel, weinorm, optimalu2_ic

m = scipy.io.loadmat('../matlab/New_res2e3_0.mat')
alp, bet = m['alpha'].item(), m['betha'].item()
t0 = time.time()
mod = OrrModel(alp, bet, N=64, Rt=2000.0, eddyfrac=0.5, units='bulk', legacy=True)
y = mod.y; cw, dw = .15, .05
yh = 1 - y[y >= 0]
wei = np.column_stack([yh, np.exp(-(yh - cw) ** 2 / 2 / dw ** 2)])
Cwei = weinorm(mod.nos, wei)
init, tfm = optimalu2_ic(mod, Cwei)
Q = mod.Q
E = lambda q: np.real(q.conj() @ Q @ q)
print('alpha, beta', alp, bet, ' tformax', tfm, ' E0', E(init))
Em = np.real(m['Ener'].ravel())
for k, t in enumerate([0.5, 1.0, 1.5]):
    q = sla.expm(mod.A * t) @ init
    print(f'  uncontrolled E({t}) python {E(q):.8f}   matlab {Em[k+2]:.8f}')
B = mod.B_legacy()
R = B.conj().T @ Q @ B * 1e5
X = sla.solve_continuous_are(mod.A, B, Q, R)
K = np.linalg.solve(R, B.conj().T @ X)
A2 = mod.A - B @ K
tl = np.linspace(0, 1.5, 5000)
Am = np.real(m['Ampli_control_t'].ravel())
for k in [1, 2, 1000, 4999]:
    q = sla.expm(A2 * tl[k]) @ init
    print(f'  LQR E(t={tl[k]:.5f}) python {E(q):.8f}  matlab {Am[k]:.8f}')
um = m['u']
print('  |u_LQR(0)| python', np.abs(K @ init), ' matlab', np.abs(um[0]))
print('  OOO (matlab LQR cost) =', m['OOO'].item(), ' fval (matlab "optimal") =', m['fval'].item())
# uncontrolled cost over [0, 1.5]
tt = np.linspace(0, 1.5, 3001)
Eh = np.array([E(sla.expm(mod.A * t) @ init) for t in tt[::100]])
w, V = np.linalg.eig(mod.A)
c = np.linalg.solve(V, init)
Et = np.array([E(V @ (np.exp(w * t) * c)) for t in tt])
print('  uncontrolled cost int_0^1.5 E dt =', np.trapezoid(Et, tt))
print('elapsed', time.time() - t0)
