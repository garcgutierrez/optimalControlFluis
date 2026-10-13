"""
Intermittent bursts observed through noisy wall-pressure sensors.

The burst is present with probability p:  z(0) = b a z0,  b ~ Bernoulli(p),  a ~ CN(0, 1).
No process noise.  Measurements y = C z + n (wall pressure at both walls), <n n^H> = V delta.
Because the burst shape is known, the information is a scalar: subtracting the (known)
response to the control, y - C z_phi = beta c(t) + n, with beta = b a and c = C z_unc(t),
and the matched-filter statistic m(t) = int c^H V^-1 (y - C z_phi) ds satisfies
    m | beta ~ CN(beta h, h),   h(t) = int_0^t c^H V^-1 c ds.
Estimators of beta:
    linear (Kalman, from the second moments):  beta_KF = p m / (1 + p h),   MSE = p/(1 + p h)
    nonlinear conditional mean:                beta_NL = pi(m) m / (1 + h),
         pi = posterior probability that the burst exists,
         pi = 1 / (1 + (1-p)/p (1+h) exp(-|m|^2/(1+h)))
Control has no dual effect, so certainty equivalence with the finite-horizon gain is optimal
(Bar-Shalom & Tse 1974) and, for any estimator used with that gain,
    J = J_FI + int_0^T w(s) MSE(s) ds,    w = rho |K_T(s) z_unc(s)|^2.
"""
import numpy as np
from scipy.integrate import quad


def wall_pressure_C(pl):
    """p(+-1) = [ i kx U'(w) v(w) + nu_T'(w) D2v(w) + nu_T(w) D3v(w) ] / k^2, from the horizontal
    momentum equations at the wall (u = w = Dv = 0 there)."""
    mod = pl.mod
    nos = mod.nos
    Z = np.zeros(nos)
    rows = []
    for w in [0, -1]:
        Up = mod.uc[w, 1]; nt = mod.nut[w, 0]; ntp = mod.nut[w, 1]
        rv = (1j * mod.alp * Up * mod.D0[w] + ntp * mod.D2[w] + nt * mod.D3[w]) / mod.k2
        rows.append(np.r_[rv, Z])
    return np.array(rows) @ np.linalg.inv(pl.L.T)


def mse_linear(p, h):
    return p / (1 + p * h)


def mse_nonlinear(p, h):
    """E|beta - E[beta|m]|^2 for the Bernoulli-Gaussian prior, by 1-D quadrature."""
    if p >= 1:
        return 1.0 / (1 + h)
    if h <= 0:
        return p
    q = (1 - p) / p * (1 + h)

    def g(r):                       # |E[beta|m]|^2 as a function of r = |m|^2
        pi = 1.0 / (1.0 + q * np.exp(-r / (1 + h)))
        return pi ** 2 * r / (1 + h) ** 2

    s1, s0 = h * (1 + h), h
    # transition of pi(r) at r* = (1+h) log(q)
    rstar = max((1 + h) * np.log(max(q, 1.0)), 0.0)
    I1 = quad(lambda x: g(s1 * x) * np.exp(-x), 0, np.inf, points=None, limit=400)[0] if rstar == 0 else \
        quad(lambda x: g(s1 * x) * np.exp(-x), 0, rstar / s1, limit=400)[0] + quad(lambda x: g(s1 * x) * np.exp(-x), rstar / s1, np.inf, limit=400)[0]
    I0 = quad(lambda x: g(s0 * x) * np.exp(-x), 0, np.inf, limit=400)[0] if rstar == 0 else \
        quad(lambda x: g(s0 * x) * np.exp(-x), 0, rstar / s0, limit=400)[0] + quad(lambda x: g(s0 * x) * np.exp(-x), rstar / s0, np.inf, limit=400)[0]
    return max(p - (p * I1 + (1 - p) * I0), 0.0)


def mse_nonlinear_mc(p, h, n=400000, seed=1):
    rng = np.random.default_rng(seed)
    b = rng.random(n) < p
    a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    beta = b * a
    m = beta * h + np.sqrt(h) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    q = (1 - p) / p * (1 + h)
    pi = 1.0 / (1.0 + q * np.exp(-np.abs(m) ** 2 / (1 + h)))
    est = pi * m / (1 + h)
    estKF = p * m / (1 + p * h)
    return np.mean(np.abs(beta - est) ** 2), np.mean(np.abs(beta - estKF) ** 2)


if __name__ == '__main__':
    for p in [1.0, 0.3, 0.1, 0.01]:
        for h in [0.1, 1.0, 10.0, 100.0, 1e4]:
            mc_nl, mc_kf = mse_nonlinear_mc(p, h)
            print(f'p={p:5g} h={h:8g}: MSE_NL quad {mse_nonlinear(p, h):.5e} MC {mc_nl:.5e} | MSE_KF {mse_linear(p, h):.5e} MC {mc_kf:.5e}')
