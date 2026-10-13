"""
Output feedback: control of the bursts from wall measurements, with sensor noise and
process noise (stochastic forcing).

Plant (energy coordinates, E = |z|^2):
    dz/dt = A z + B phi + w,        <w w^H> = W delta(t - t')
    y     = C z + n,                <n n^H> = V delta(t - t')
    z(0) ~ CN(0, Sigma0)            (burst of random amplitude and phase)
Cost:   J = E int_0^T (|z|^2 + rho |phi|^2) dt

Compensators (all with estimate zhat(0) = E[z(0)] = 0):
  'LQG'    : stationary Kalman gain L (designed for a fictitious W_d and V) + LQR gain K
  'KF+LQR' : time-varying Kalman-Bucy filter started from Sigma0 + LQR gain
  'opt'    : time-varying Kalman-Bucy filter + finite-horizon gain P_T(t).  By the separation
             theorem (Wonham 1968) this is the optimum over all causal controllers.

Expected costs (exact; no Monte Carlo):
  full information      J_FI  = tr(Pi(T) Sigma0) + int_0^T tr(Pi(T-s) W) ds
  optimal filter + gain J     = J_FI + int_0^T tr(Sigma(s) X(T-s)) ds
      X = Po S Po                   for the finite-horizon gain (Po = P_T),
      X = PL S P + P S PL - P S P   for the LQR gain (PL = LQR cost-to-go over T-s),
  obtained by integrating by parts the innovation form of the cost with the filter
  Riccati equation, which removes the singular behaviour of the innovations near s = 0.
  Stationary LQG: covariance of the augmented state (z, e), block-wise Lyapunov/Sylvester.
"""
import numpy as np
import scipy.linalg as sla

from orrss import herm
from lq import lyap


# ----------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------
def wall_shear_C(pl):
    """Streamwise and spanwise wall shear stress at both walls, in z coordinates.
    tau_x = du/dy = i(alp D2 v - bet D eta)/k^2,  tau_z = dw/dy = i(bet D2 v + alp D eta)/k^2."""
    mod = pl.mod
    D1, D2 = mod.D1, mod.D2
    k2, a, b = mod.k2, mod.alp, mod.bet
    rows = []
    for w in [0, -1]:                       # y = +1, y = -1
        rows.append(np.r_[1j * a * D2[w], -1j * b * D1[w]] / k2)
        rows.append(np.r_[1j * b * D2[w], 1j * a * D1[w]] / k2)
    Cq = np.array(rows)
    return Cq @ np.linalg.inv(pl.L.T)       # z = L^T q


# ----------------------------------------------------------------------------
# Riccati equations
# ----------------------------------------------------------------------------
def care_general(A, S, Q, newton=3):
    """Stabilising solution of A^H P + P A - P S P + Q = 0."""
    n = A.shape[0]
    H = np.block([[A, -S], [-Q, -A.conj().T]])
    T, Z, sdim = sla.schur(H, output='complex', sort='lhp')
    if sdim != n:
        raise RuntimeError('Hamiltonian has eigenvalues on the imaginary axis')
    X1, X2 = Z[:n, :n], Z[n:, :n]
    P = herm(np.linalg.solve(X1.conj().T, X2.conj().T).conj().T)
    for _ in range(newton):
        Ac = A - S @ P
        P = herm(sla.solve_continuous_lyapunov(Ac.conj().T, -(Q + P @ S @ P)))
    return P


def kalman_stationary(A, C, W, V):
    """Stationary Kalman-Bucy covariance: A S + S A^H - S C^H V^-1 C S + W = 0."""
    Sf = herm(C.conj().T @ np.linalg.solve(V, C))
    if np.linalg.norm(W) == 0:
        Sig = np.zeros_like(A)            # A is stable
    else:
        Sig = care_general(A.conj().T, Sf, herm(W))
    L = Sig @ C.conj().T @ np.linalg.inv(V)
    return Sig, L, Sf


class Filter:
    """Time-varying Kalman-Bucy covariance from Sigma(0) = Sigma0, in closed form:
        Sigma(t) = Sigma_inf + e^{Af t} D0 [I + H(t) D0]^{-1} e^{Af^H t},  D0 = Sigma0 - Sigma_inf,
        H(t) = int_0^t e^{Af^H s} Sf e^{Af s} ds."""

    def __init__(self, A, C, W, V, Sigma0, expA=None):
        self.Sinf, self.Linf, self.Sf = kalman_stationary(A, C, W, V)
        self.Af = A - self.Sinf @ self.Sf
        self.Hinf = lyap(self.Af.conj().T, self.Sf)
        self.D0 = herm(Sigma0 - self.Sinf)
        self.n = A.shape[0]
        self.noW = np.linalg.norm(W) == 0
        self.expA = expA                    # precomputed e^{A s} (valid when W = 0)

    def sigma(self, t, k=None):
        if self.noW and self.expA is not None and k is not None:
            E = self.expA[k]
        else:
            E = sla.expm(self.Af * t)
        H = herm(self.Hinf - E.conj().T @ self.Hinf @ E)
        Y = self.D0 @ np.linalg.inv(np.eye(self.n) + H @ self.D0)
        return herm(self.Sinf + E @ Y @ E.conj().T)


# ----------------------------------------------------------------------------
# Quantities that depend only on the horizon
# ----------------------------------------------------------------------------
def quad_nodes(T, n=60, smin=1e-6):
    """Union of geometric nodes clustered near s = 0 (fast assimilation of the first
    measurements), near s = T (fast decay of the cost-to-go of the stiff modes), and
    uniform nodes (process noise acts during the whole horizon)."""
    g = T * np.geomspace(smin, 0.5, n // 3)
    s = np.unique(np.r_[0.0, g, T - g, np.linspace(0, T, n // 3 + 1), T])
    w = np.zeros_like(s)
    w[1:] += 0.5 * np.diff(s); w[:-1] += 0.5 * np.diff(s)
    return s, w


class Horizon:
    """Cost-to-go matrices at the quadrature nodes, for a given LQProblem and T."""

    def __init__(self, lq, T, nq=60, keep_expA=True):
        self.lq, self.T = lq, T
        A, n, P, Ac, Ginf, S = lq.A, lq.n, lq.P, lq.Ac, lq.Ginf, lq.S
        I = np.eye(n)
        self.s, self.w = quad_nodes(T, nq)
        self.PL, self.Po, self.W0, self.XL, self.XO, self.expA = [], [], [], [], [], []
        PSP = herm(P @ S @ P)
        for sk in self.s:
            tau = T - sk
            Ec = sla.expm(Ac * tau)
            PL = herm(P - Ec.conj().T @ P @ Ec)
            GT = herm(Ginf - Ec @ Ginf @ Ec.conj().T)
            X = np.linalg.solve(I - GT @ P, GT @ P @ Ec)
            Po = herm(PL - Ec.conj().T @ P @ X)
            E0 = sla.expm(A * tau)
            W0 = herm(lq.Winf - E0.conj().T @ lq.Winf @ E0)
            self.PL.append(PL); self.Po.append(Po); self.W0.append(W0)
            self.XL.append(herm(PL @ S @ P + P @ S @ PL - PSP))
            self.XO.append(herm(Po @ S @ Po))
            if keep_expA:
                self.expA.append(sla.expm(A * sk))

    def full_information(self, Sigma0, W, Zunc=None, Zfi=None):
        """Zunc, Zfi: covariance of the background state at t = 0 (statistical equilibrium under the
        forcing W without control, and with the full-information LQR, respectively)."""
        tr = lambda M, X: float(np.real(np.sum(M * X.T)))       # tr(M X)
        w = self.w
        Zunc = 0 * Sigma0 if Zunc is None else Zunc
        Zfi = 0 * Sigma0 if Zfi is None else Zfi
        return dict(unc=tr(self.W0[0], Sigma0 + Zunc) + sum(wk * tr(M, W) for wk, M in zip(w, self.W0)),
                    FI_LQR=tr(self.PL[0], Sigma0 + Zfi) + sum(wk * tr(M, W) for wk, M in zip(w, self.PL)),
                    FI_opt=tr(self.Po[0], Sigma0 + Zfi) + sum(wk * tr(M, W) for wk, M in zip(w, self.Po)))

    def output_feedback(self, C, W, V, Sigma0, Zunc=None, Zfi=None, Zlqg=None):
        """The time-varying compensators are assumed to have been operating as the standard stationary
        LQG before the burst is born at t = 0 (state covariance Zlqg, filter covariance Sigma_inf); they
        restart the filter from Sigma_inf + Sigma0 when the burst appears."""
        c = self.full_information(Sigma0, W, Zunc, Zfi)
        flt = Filter(self.lq.A, C, W, V, Sigma0, expA=self.expA if self.expA else None)
        flt.D0 = herm(Sigma0)                       # Sigma(0) = Sigma_inf + Sigma0
        Zlqg = 0 * Sigma0 if Zlqg is None else Zlqg
        tr = lambda M, X: float(np.real(np.sum(M * X.T)))
        base_L = tr(self.PL[0], Sigma0 + Zlqg) + sum(wk * tr(M, W) for wk, M in zip(self.w, self.PL))
        base_O = tr(self.Po[0], Sigma0 + Zlqg) + sum(wk * tr(M, W) for wk, M in zip(self.w, self.Po))
        trS = []; eL = []; eO = []
        for k, sk in enumerate(self.s):
            Sig = flt.sigma(sk, k)
            trS.append(float(np.real(np.trace(Sig))))
            eL.append(float(np.real(np.sum(Sig * self.XL[k].T))))
            eO.append(float(np.real(np.sum(Sig * self.XO[k].T))))
        c['KF_LQR'] = base_L + self.w @ np.array(eL)
        c['opt'] = base_O + self.w @ np.array(eO)
        c['trSigma'] = np.array(trS)
        c['s'] = self.s
        return c


# ----------------------------------------------------------------------------
# Stationary (time-invariant) LQG
# ----------------------------------------------------------------------------
class StationaryLQG:
    """Expected cost of zhat' = A zhat + B phi + L (y - C zhat), phi = -K zhat, from the covariance
    of (z, e).  F = [[Ac, BK], [0, Af]] is block triangular, and the Lyapunov equations for the
    cost matrices are solved block by block."""

    def __init__(self, lq, T):
        self.lq, self.T = lq, T
        A, B, K, n = lq.A, lq.B, lq.K, lq.n
        self.BK = B @ K
        self.KK = lq.rho * K.conj().T @ K
        # Y11 = P (LQR identity), Z11 solves Ac^H Z11 + Z11 Ac + P = 0
        self.Y11 = lq.P
        self.Z11 = herm(sla.solve_continuous_lyapunov(lq.Ac.conj().T, -lq.P))

    def cost(self, C, W, V, Sigma0, L):
        lq, T, n = self.lq, self.T, self.lq.n
        Ac, BK, KK = lq.Ac, self.BK, self.KK
        Af = lq.A - L @ C
        AcH, AfH = Ac.conj().T, Af.conj().T
        # Y = int_0^inf e^{F^H t} M e^{F t} dt,  M = [[I+KK, -KK], [-KK, KK]]
        Y11 = self.Y11
        Y12 = sla.solve_sylvester(AcH, Af, -(Y11 @ BK - KK))
        Y22 = herm(sla.solve_continuous_lyapunov(AfH, -(KK + BK.conj().T @ Y12 + Y12.conj().T @ BK)))
        Y = np.block([[Y11, Y12], [Y12.conj().T, Y22]])
        F = np.block([[Ac, BK], [np.zeros((n, n)), Af]])
        EF = sla.expm(F * T)
        YT = herm(Y - EF.conj().T @ Y @ EF)
        X0 = np.block([[Sigma0, Sigma0], [Sigma0, Sigma0]])
        LVL = L @ V @ L.conj().T
        if np.linalg.norm(W) > 0:
            X0 = X0 + self.stationary_covariance(W, LVL, Af)
        J = float(np.real(np.sum(YT * X0.T)))
        if np.linalg.norm(W) > 0 or np.linalg.norm(LVL) > 0:
            Z11 = self.Z11
            Z12 = sla.solve_sylvester(AcH, Af, -(Z11 @ BK + Y12))
            Z22 = herm(sla.solve_continuous_lyapunov(AfH, -(Y22 + BK.conj().T @ Z12 + Z12.conj().T @ BK)))
            Z = np.block([[Z11, Z12], [Z12.conj().T, Z22]])
            intY = T * Y - herm(Z - EF.conj().T @ Z @ EF)
            Naug = np.block([[W, W], [W, W + LVL]])
            J += float(np.real(np.sum(intY * Naug.T)))
        return J

    def stationary_covariance(self, W, LVL, Af):
        """Covariance of (z, e) in statistical equilibrium: F X + X F^H + Naug = 0, block by block."""
        Ac, BK = self.lq.Ac, self.BK
        Xee = herm(sla.solve_continuous_lyapunov(Af, -(W + LVL)))
        Xze = sla.solve_sylvester(Ac, Af.conj().T, -(BK @ Xee + W))
        Xzz = herm(sla.solve_continuous_lyapunov(Ac, -(BK @ Xze.conj().T + Xze @ BK.conj().T + W)))
        return np.block([[Xzz, Xze], [Xze.conj().T, Xee]])

    def best(self, C, W, V, Sigma0, Wshape, omegas):
        """Tune a fictitious process noise omega*Wshape in the design of the filter."""
        Js = []
        for om in omegas:
            _, L, _ = kalman_stationary(self.lq.A, C, W + om * Wshape, V)
            Js.append(self.cost(C, W, V, Sigma0, L))
        Js = np.array(Js)
        k = int(np.argmin(Js))
        return Js[k], omegas[k], Js


def output_power(pl, C, z0, tmax):
    """Mean squared wall signal of the uncontrolled burst over [0, tmax], per sensor."""
    Wo = lyap(pl.A.conj().T, C.conj().T @ C)
    E = sla.expm(pl.A * tmax)
    return float(np.real(z0.conj() @ (Wo - E.conj().T @ Wo @ E) @ z0)) / (C.shape[0] * tmax)
