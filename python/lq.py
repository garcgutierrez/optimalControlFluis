"""
Linear-quadratic control of the Orr--Sommerfeld--Squire model.

All computations are done in energy coordinates z = L^H q (Q = L L^H), so that
the kinetic energy is E = |z|^2.  The plant is dz/dt = A z + B u.

Cost over a finite horizon T (the lifetime of the linear burst):
    J_T[u] = int_0^T ( |z|^2 + rho |u|^2 ) dt

Implemented controllers
  * LQR: time-invariant linear state feedback u = -K z, K = B^H P / rho
         (optimal for T -> infinity).
  * Finite-horizon optimum: the unrestricted minimiser of J_T.  It is computed
         (i) in closed form from the Hamiltonian system (P, G Gramians), and
         (ii) by direct quasi-Newton (L-BFGS) optimisation of the control history,
         which does not assume any form of the control law.
  * Bounded actuation |u_j| <= umax: constrained optimum by accelerated projected
         gradient (FISTA) and saturated LQR.
"""
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from orrss import OrrModel, herm


# ----------------------------------------------------------------------------
# System assembly
# ----------------------------------------------------------------------------
class Plant:
    def __init__(self, alp, bet, N=128, Rt=2000.0, eddyfrac=1.0, actuator='wall', profile=None, Rbc=None):
        mod = OrrModel(alp, bet, N=N, Rt=Rt, eddyfrac=eddyfrac, units='tau', profile=profile, Rbc=Rbc)
        self.mod = mod
        self.alp, self.bet = alp, bet
        Q = mod.Q
        L = np.linalg.cholesky(Q)             # Q = L L^T (real)
        self.L = L
        Linv = np.linalg.inv(L)
        self.A = L.T @ mod.A @ Linv.T          # z = L^T q
        Phi = mod.physical_basis()             # energy-orthonormal basis of BC subspace
        self.Phi = Phi
        self.Zphys = L.T @ Phi                 # orthonormal columns in z-space
        self.AS = Phi.conj().T @ Q @ mod.A @ Phi   # dynamics restricted to the physical subspace
        if actuator == 'wall':
            self.B = L.T @ mod.B_wall()
        elif actuator == 'distributed':
            self.B = self.Zphys.copy()
        else:
            raise ValueError(actuator)
        self.actuator = actuator
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

    def to_q(self, z):
        return np.linalg.solve(self.L.T, z)

    # optimal transient growth restricted to physical initial conditions
    def transient_growth(self, tgrid=None, tmax_guess=None):
        AS = self.AS
        w, V = np.linalg.eig(AS)
        Vi = np.linalg.inv(V)

        def G(t):
            M = (V * np.exp(w * t)) @ Vi
            return np.linalg.norm(M, 2) ** 2

        if tgrid is None:
            tgrid = np.geomspace(1e-4, 10.0, 120)
        g = np.array([G(t) for t in tgrid])
        i = int(np.argmax(g))
        # refine with golden section on log t
        from scipy.optimize import minimize_scalar
        lo = tgrid[max(i - 1, 0)]; hi = tgrid[min(i + 1, len(tgrid) - 1)]
        if i == 0:
            return 0.0, 1.0, self.Zphys[:, 0] * 0, tgrid, g
        res = minimize_scalar(lambda s: -G(np.exp(s)), bounds=(np.log(lo), np.log(hi)),
                              method='bounded', options={'xatol': 1e-6})
        tmax = float(np.exp(res.x)); Gmax = -res.fun
        M = (V * np.exp(w * tmax)) @ Vi
        U, S, Vh = np.linalg.svd(M)
        xi0 = Vh.conj().T[:, 0]
        z0 = self.Zphys @ xi0
        return tmax, Gmax, z0, tgrid, g


# ----------------------------------------------------------------------------
# Gramians and Riccati solutions
# ----------------------------------------------------------------------------
def worst_case_ratio(ML, Gap, rel=1e-10):
    """max over initial conditions of J_LQR/J_opt, with J_opt = J_LQR - Gap
    (both restricted to the physical subspace).  Stable whitened formulation."""
    lam, V = np.linalg.eigh(herm(ML))
    keep = lam > rel * lam.max()
    W = V[:, keep] / np.sqrt(lam[keep])
    mu = np.linalg.eigvalsh(herm(W.conj().T @ Gap @ W))
    mu = np.clip(mu.max(), 0.0, 1 - 1e-15)
    return 1.0 / (1.0 - mu)


def lyap(A, Qm):
    """Solve A X + X A^H + Qm = 0."""
    return herm(sla.solve_continuous_lyapunov(A, -Qm))


def care(A, B, rho, newton_steps=3):
    """Stabilising solution of A^H P + P A - P B B^H P / rho + I = 0.
    Ordered Schur decomposition of the Hamiltonian matrix followed by a few
    Newton--Kleinman refinement steps (much faster than the QZ-based solver of
    scipy for the stiff operators considered here, same accuracy)."""
    n = A.shape[0]
    S = B @ B.conj().T / rho
    H = np.block([[A, -S], [-np.eye(n), -A.conj().T]])
    T, Z, sdim = sla.schur(H, output='complex', sort='lhp')
    if sdim != n:
        raise RuntimeError('Hamiltonian splitting failed')
    X1, X2 = Z[:n, :n], Z[n:, :n]
    P = herm(np.linalg.solve(X1.conj().T, X2.conj().T).conj().T)
    for _ in range(newton_steps):
        K = B.conj().T @ P / rho
        Ac = A - B @ K
        P = herm(sla.solve_continuous_lyapunov(Ac.conj().T, -(np.eye(n) + rho * K.conj().T @ K)))
    return P


def lqr(A, B, rho):
    P = care(A, B, rho)
    K = B.conj().T @ P / rho
    return P, K


class LQProblem:
    """Everything needed to compare LQR and the finite-horizon optimum."""

    def __init__(self, A, B, rho):
        self.A, self.B, self.rho = A, B, rho
        n = A.shape[0]
        self.n = n
        self.P, self.K = lqr(A, B, rho)
        self.Ac = A - B @ self.K
        self.S = B @ B.conj().T / rho
        self.Ginf = lyap(self.Ac, self.S)            # closed-loop controllability Gramian
        self.Winf = lyap(A.conj().T, np.eye(n))      # uncontrolled cost matrix (T = inf)

    # --- matrices defining the costs over [0, T] -----------------------------
    def cost_matrices(self, T):
        A, Ac, P, n = self.A, self.Ac, self.P, self.n
        E0 = sla.expm(A * T)
        Ec = sla.expm(Ac * T)
        W0 = herm(self.Winf - E0.conj().T @ self.Winf @ E0)        # uncontrolled
        PL = herm(P - Ec.conj().T @ P @ Ec)                         # LQR over [0,T]
        GT = herm(self.Ginf - Ec @ self.Ginf @ Ec.conj().T)         # finite-time Gramian
        # optimality gap of LQR: x_T^H P (I - G_T P)^{-1} G_T P x_T,  x_T = Ec z0
        X = np.linalg.solve(np.eye(n) - GT @ P, GT @ P @ Ec)
        Gap = herm(Ec.conj().T @ P @ X)
        Popt = herm(PL - Gap)                                       # finite-horizon optimum
        return dict(W0=W0, PL=PL, Popt=Popt, Gap=Gap, GT=GT, Ec=Ec, E0=E0)

    def costs(self, z0, T):
        cm = self.cost_matrices(T)
        f = lambda M: float(np.real(z0.conj() @ M @ z0))
        return f(cm['W0']), f(cm['PL']), f(cm['Popt']), cm

    # --- trajectories ----------------------------------------------------------
    def lqr_trajectory(self, z0, t):
        w, V = np.linalg.eig(self.Ac)
        c = np.linalg.solve(V, z0)
        Z = (V[None, :, :] * np.exp(np.outer(t, w))[:, None, :]) @ c
        U = -(self.K @ Z.T).T
        return Z, U

    def uncontrolled_trajectory(self, z0, t):
        w, V = np.linalg.eig(self.A)
        c = np.linalg.solve(V, z0)
        Z = (V[None, :, :] * np.exp(np.outer(t, w))[:, None, :]) @ c
        return Z

    def optimal_trajectory(self, z0, T, t):
        """Finite-horizon optimum from the Hamiltonian system:
            u = -K z - R^{-1} B^H exp(Ac^H (T-t)) e_T,
            z = exp(Ac t) z0 - G(t) exp(Ac^H (T-t)) e_T,
            e_T = -(I - P G_T)^{-1} P exp(Ac T) z0."""
        n, P, Ac, Ginf = self.n, self.P, self.Ac, self.Ginf
        Ec = sla.expm(Ac * T)
        GT = herm(Ginf - Ec @ Ginf @ Ec.conj().T)
        eT = -np.linalg.solve(np.eye(n) - P @ GT, P @ Ec @ z0)
        Z = np.zeros((len(t), n), complex)
        Uc = np.zeros((len(t), self.B.shape[1]), complex)
        for k, tk in enumerate(t):
            Et = sla.expm(Ac * tk)
            Gt = herm(Ginf - Et @ Ginf @ Et.conj().T)
            et = sla.expm(Ac.conj().T * (T - tk)) @ eT
            Z[k] = Et @ z0 - Gt @ et
            lam = P @ Z[k] + et
            Uc[k] = -(self.B.conj().T @ lam) / self.rho
        return Z, Uc


# ----------------------------------------------------------------------------
# Direct optimisation of the control history (no assumption on the control law)
# ----------------------------------------------------------------------------
class Discretised:
    """Zero-order-hold discretisation on Nt intervals of [0, T]; trapezoidal
    quadrature for the energy term."""

    def __init__(self, A, B, T, Nt):
        n, m = B.shape
        h = T / Nt
        M = np.zeros((n + m, n + m), complex)
        M[:n, :n] = A * h
        M[:n, n:] = B * h
        EM = sla.expm(M)
        self.Eh = EM[:n, :n]
        self.Gh = EM[:n, n:]
        self.h, self.Nt, self.n, self.m, self.T = h, Nt, n, m, T
        w = np.full(Nt + 1, h); w[0] = w[-1] = h / 2
        self.w = w

    def simulate(self, z0, U):
        Z = np.zeros((self.Nt + 1, self.n), complex)
        Z[0] = z0
        for k in range(self.Nt):
            Z[k + 1] = self.Eh @ Z[k] + self.Gh @ U[k]
        return Z

    def cost_grad(self, z0, U, rho):
        Z = self.simulate(z0, U)
        J = float(np.sum(self.w * np.sum(np.abs(Z) ** 2, axis=1)) + rho * self.h * np.sum(np.abs(U) ** 2))
        lam = self.w[-1] * Z[-1]
        gU = np.zeros_like(U)
        EhH = self.Eh.conj().T
        GhH = self.Gh.conj().T
        for k in range(self.Nt - 1, -1, -1):
            gU[k] = GhH @ lam + rho * self.h * U[k]
            lam = self.w[k] * Z[k] + EhH @ lam
        # gradient w.r.t. real and imaginary parts: dJ/dRe = 2 Re(g), dJ/dIm = 2 Im(g)
        return J, 2 * gU, Z


def direct_optimisation(disc, z0, rho, U0=None, maxiter=2000, gtol=1e-10, history=False, method='L-BFGS-B'):
    """Quasi-Newton (L-BFGS) minimisation of J over the piecewise-constant control
    history.  Nothing is assumed about the control law."""
    Nt, m = disc.Nt, disc.m
    if U0 is None:
        U0 = np.zeros((Nt, m), complex)
    hist = []

    def unpack(x):
        return (x[:Nt * m] + 1j * x[Nt * m:]).reshape(Nt, m)

    def fun(x):
        U = unpack(x)
        J, g, _ = disc.cost_grad(z0, U, rho)
        if history:
            hist.append(J)
        g = g.ravel()
        return J, np.concatenate([g.real, g.imag])

    x0 = np.concatenate([U0.real.ravel(), U0.imag.ravel()])
    res = minimize(fun, x0, jac=True, method=method,
                   options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxcor=30))
    U = unpack(res.x)
    return U, res.fun, hist, res


def discrete_riccati_optimum(disc, z0, rho):
    """Exact minimiser of the discretised problem (backward Riccati recursion);
    used to check the convergence of the direct optimisation."""
    n, m, Nt, h = disc.n, disc.m, disc.Nt, disc.h
    E, G, w = disc.Eh, disc.Gh, disc.w
    S = w[-1] * np.eye(n)
    Ks = [None] * Nt
    for k in range(Nt - 1, -1, -1):
        Rk = rho * h * np.eye(m) + G.conj().T @ S @ G
        Kk = np.linalg.solve(Rk, G.conj().T @ S @ E)
        Ks[k] = Kk
        S = w[k] * np.eye(n) + E.conj().T @ S @ (E - G @ Kk)
        S = herm(S)
    J = float(np.real(z0.conj() @ S @ z0))
    Z = np.zeros((Nt + 1, n), complex); U = np.zeros((Nt, m), complex)
    Z[0] = z0
    for k in range(Nt):
        U[k] = -Ks[k] @ Z[k]
        Z[k + 1] = E @ Z[k] + G @ U[k]
    return U, J, Z


# ----------------------------------------------------------------------------
# Bounded actuation
# ----------------------------------------------------------------------------
def project_disk(U, umax):
    a = np.abs(U)
    f = np.minimum(1.0, umax / np.maximum(a, 1e-300))
    return U * f


def constrained_optimum(disc, z0, rho, umax, U0=None, maxiter=20000, tol=1e-12, L=None):
    """FISTA (accelerated projected gradient) for min J s.t. |u_j(t)| <= umax.
    The problem is convex, so the iteration converges to the global optimum."""
    Nt, m = disc.Nt, disc.m
    if U0 is None:
        U0 = np.zeros((Nt, m), complex)
    if L is None:
        L = lipschitz(disc, z0, rho)
    X = project_disk(U0, umax); Y = X.copy(); tk = 1.0
    Jold = np.inf
    for it in range(maxiter):
        J, g, _ = disc.cost_grad(z0, Y, rho)
        g = 0.5 * g   # Wirtinger gradient dJ/dU*
        Xn = project_disk(Y - g / L, umax)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * tk ** 2))
        Y = Xn + (tk - 1) / tn * (Xn - X)
        X, tk = Xn, tn
        if it % 50 == 0:
            Jx = disc.cost_grad(z0, X, rho)[0]
            if abs(Jold - Jx) < tol * abs(Jx):
                break
            Jold = Jx
    J = disc.cost_grad(z0, X, rho)[0]
    return X, J, it


def lipschitz(disc, z0, rho, iters=60):
    """Largest eigenvalue of the Hessian of J (w.r.t. U, Wirtinger scaling) by power iteration."""
    Nt, m = disc.Nt, disc.m
    rng = np.random.default_rng(0)
    V = rng.standard_normal((Nt, m)) + 1j * rng.standard_normal((Nt, m))
    zero = np.zeros_like(z0)
    lam = 1.0
    for _ in range(iters):
        V /= np.linalg.norm(V)
        _, g, _ = disc.cost_grad(zero, V, rho)
        HV = 0.5 * g
        lam = np.real(np.vdot(V, HV))
        V = HV
    return 1.05 * lam


def saturated_lqr(disc, z0, K, rho, umax):
    """Closed-loop simulation of u = sat(-K z) with ZOH on the grid of disc."""
    Nt = disc.Nt
    Z = np.zeros((Nt + 1, disc.n), complex); U = np.zeros((Nt, disc.m), complex)
    Z[0] = z0
    for k in range(Nt):
        U[k] = project_disk(-(K @ Z[k])[None, :], umax)[0]
        Z[k + 1] = disc.Eh @ Z[k] + disc.Gh @ U[k]
    J = float(np.sum(disc.w * np.sum(np.abs(Z) ** 2, axis=1)) + rho * disc.h * np.sum(np.abs(U) ** 2))
    return U, J, Z


# ----------------------------------------------------------------------------
# Random (exponentially distributed) lifetime: discounted cost
# ----------------------------------------------------------------------------
def discounted_costs(A, B, rho, K, z0, mu, Pmu=None):
    """Expected cost E_T[J_T] for T ~ Exp(mu) = int_0^inf exp(-mu t)(|z|^2+rho|u|^2) dt.
    Returns (J uncontrolled, J with the undiscounted LQR gain K, J optimal) and the
    optimal (discounted) gain.  The optimum is the LQR of the shifted operator
    A - mu/2 I, i.e. again a linear time-invariant feedback."""
    n, m = B.shape
    I = np.eye(n)
    As = A - 0.5 * mu * I
    W = lyap(As.conj().T, I)
    Acs = As - B @ K
    PL = herm(sla.solve_continuous_lyapunov(Acs.conj().T, -(I + rho * K.conj().T @ K)))
    if Pmu is None:
        Pmu = care(As, B, rho)
    Kmu = B.conj().T @ Pmu / rho
    f = lambda M: float(np.real(z0.conj() @ M @ z0))
    return f(W), f(PL), f(Pmu), Kmu, dict(W=W, PL=PL, Pmu=Pmu)
