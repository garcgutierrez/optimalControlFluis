"""
Drag objective with a non-quadratic (amplitude-proportional) actuation cost, full information.

Drag increment of a burst (FIK identity at constant flow rate): D(t) = int_{-1}^{1} y Re(u v*) dy,
an indefinite quadratic form z^H M_D z.  Cost for a burst of complex amplitude a (z(0) = a z0):
    J[phi] = int_0^T ( D + rho |phi|^2 + gamma |phi|_1 ) dt,
with |phi|_1 = sum_j |phi_j| (one term per actuator; for the distributed forcing, the Euclidean
norm of the forcing, i.e. a group penalty).  With phi = a psi, J = |a|^2 [Q(psi) + (gamma/|a|) |psi|_1],
so the optimum for amplitude |a| follows from the value function
    V(g) = min_psi Q(psi) + g |psi|_1 .
For a ~ CN(0,1):
    nonlinear optimum          E J = int_0^inf r^2 V(gamma/r) 2 r exp(-r^2) dr
    best amplitude-linear law  E J = V(gamma E|a|),  E|a| = sqrt(pi)/2      (phi = a psi, psi fixed)
"""
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh


def cc_weights(M):
    """Clenshaw-Curtis nodes and weights on M+1 Gauss-Lobatto points."""
    th = np.pi * np.arange(M + 1) / M
    x = np.cos(th)
    w = np.zeros(M + 1)
    v = np.ones(M - 1)
    if M % 2 == 0:
        w[0] = w[M] = 1.0 / (M ** 2 - 1)
        for k in range(1, M // 2):
            v -= 2 * np.cos(2 * k * th[1:-1]) / (4 * k ** 2 - 1)
        v -= np.cos(M * th[1:-1]) / (M ** 2 - 1)
    else:
        w[0] = w[M] = 1.0 / M ** 2
        for k in range(1, (M - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * th[1:-1]) / (4 * k ** 2 - 1)
    w[1:-1] = 2 * v / M
    return x, w


def drag_matrix(pl, Mfine=None):
    """M_D (z coordinates) with D = z^H M_D z = int y Re(u v*) dy, evaluated exactly with a
    Clenshaw-Curtis rule on a grid fine enough for the product of two polynomials of degree N."""
    mod = pl.mod
    nos = mod.nos
    N = nos - 1
    Mf = Mfine or (2 * N + 4)
    x, wq = cc_weights(Mf)
    th = np.arccos(np.clip(x, -1, 1))
    Tf = np.cos(np.outer(th, np.arange(nos)))           # T_j(x_i)
    Dc = mod.Dc                                          # coefficients of the derivative
    Vq = np.hstack([Tf, np.zeros((len(x), nos))])
    Uq = 1j * np.hstack([mod.alp * Tf @ Dc, -mod.bet * Tf]) / mod.k2
    Wy = wq * x
    Mq = 0.5 * (Uq.conj().T @ (Wy[:, None] * Vq) + Vq.conj().T @ (Wy[:, None] * Uq))
    Li = np.linalg.inv(pl.L)
    return 0.5 * (Li @ Mq @ Li.T + (Li @ Mq @ Li.T).conj().T)


class DragProblem:
    def __init__(self, disc, z0, M, rho, group):
        """disc: lq.Discretised (ZOH on [0, T]); M: state weight; group: 'input' (one L1 term per
        actuator) or 'vector' (Euclidean norm of the input vector at each time)."""
        self.d, self.z0, self.M, self.rho, self.group = disc, z0, M, rho, group

    def smooth(self, U, z0=None):
        d = self.d
        z0 = self.z0 if z0 is None else z0
        Z = np.zeros((d.Nt + 1, d.n), complex); Z[0] = z0
        for k in range(d.Nt):
            Z[k + 1] = d.Eh @ Z[k] + d.Gh @ U[k]
        MZ = Z @ self.M.T
        q = np.real(np.sum(Z.conj() * MZ, 1))
        J = float(np.sum(d.w * q) + self.rho * d.h * np.sum(np.abs(U) ** 2))
        lam = d.w[-1] * MZ[-1]
        g = np.zeros_like(U)
        EhH, GhH = d.Eh.conj().T, d.Gh.conj().T
        for k in range(d.Nt - 1, -1, -1):
            g[k] = GhH @ lam + self.rho * d.h * U[k]
            lam = d.w[k] * MZ[k] + EhH @ lam
        return J, g, Z                                   # g = dJ/dU* (Wirtinger)

    def l1(self, U):
        if self.group == 'input':
            return self.d.h * float(np.sum(np.abs(U)))
        return self.d.h * float(np.sum(np.linalg.norm(U, axis=1)))

    def prox(self, U, tau):
        if self.group == 'input':
            a = np.abs(U)
            return U * np.maximum(0.0, 1.0 - tau / np.maximum(a, 1e-300))
        a = np.linalg.norm(U, axis=1)
        return U * np.maximum(0.0, 1.0 - tau / np.maximum(a, 1e-300))[:, None]

    def hessian_extremes(self):
        """Largest and smallest eigenvalues of the Hessian of the smooth part (Wirtinger scaling)."""
        d = self.d
        shp = (d.Nt, d.m)
        zero = np.zeros_like(self.z0)

        def mv(x):
            U = (x[:d.Nt * d.m] + 1j * x[d.Nt * d.m:]).reshape(shp)
            _, g, _ = self.smooth(U, zero)
            g = g.ravel()
            return np.concatenate([g.real, g.imag])
        n2 = 2 * d.Nt * d.m
        op = LinearOperator((n2, n2), matvec=mv, dtype=float)
        lmax = eigsh(op, k=1, which='LA', tol=1e-4, return_eigenvectors=False)[0]
        lmin = eigsh(op, k=1, which='SA', tol=1e-4, return_eigenvectors=False)[0]
        return lmax, lmin

    def threshold(self):
        """Smallest L1 weight for which psi = 0 is optimal."""
        _, g, _ = self.smooth(np.zeros((self.d.Nt, self.d.m), complex))
        if self.group == 'input':
            return 2 * np.max(np.abs(g)) / self.d.h
        return 2 * np.max(np.linalg.norm(g, axis=1)) / self.d.h

    def solve(self, gam, L, U0=None, maxiter=20000, tol=1e-10):
        """FISTA for min Q(U) + gam * l1(U) (convex when the Hessian is positive definite)."""
        d = self.d
        X = np.zeros((d.Nt, d.m), complex) if U0 is None else U0.copy()
        Y = X.copy(); tk = 1.0; Jold = np.inf
        tau = gam * d.h / (2 * L)
        for it in range(maxiter):
            _, g, _ = self.smooth(Y)
            Xn = self.prox(Y - g / L, tau)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk ** 2))
            Y = Xn + (tk - 1) / tn * (Xn - X)
            X, tk = Xn, tn
            if it % 25 == 0:
                Jx = self.smooth(X)[0] + gam * self.l1(X)
                if Jx > Jold + 1e-14 * abs(Jold):      # adaptive restart
                    Y = X.copy(); tk = 1.0
                if abs(Jold - Jx) < tol * max(abs(Jx), 1e-30):
                    break
                Jold = Jx
        Q = self.smooth(X)[0]
        return X, Q, self.l1(X), it


def expected_costs(gams, Vgrid, Vfun):
    """Nonlinear optimum and best amplitude-linear law, for a ~ CN(0,1)."""
    from scipy.integrate import quad
    out_nl, out_lin = [], []
    for gam in gams:
        f = lambda r: r ** 2 * Vfun(gam / r) * 2 * r * np.exp(-r ** 2) if r > 0 else 0.0
        out_nl.append(quad(f, 0, 8, limit=400)[0])
        out_lin.append(Vfun(gam * np.sqrt(np.pi) / 2))
    return np.array(out_nl), np.array(out_lin)


def make_Vfun(d):
    """Interpolant of the value function V(g) (log-linear in g), V = D0 above the threshold."""
    gs, V, gc, D0 = d['gs'], d['V'], d['gc'], d['D0']
    lg = np.log(gs[:-1]); Vv = V[:-1]; V0 = V[-1]; gmin = gs[-2]

    def Vfun(g):
        if g >= gc:
            return D0
        if g <= gmin:
            return V0 + (Vv[-1] - V0) * g / gmin
        return float(np.interp(np.log(g), lg[::-1], Vv[::-1]))
    return Vfun


AMPLITUDES = {
    # name: (density of r = |a| with E r^2 = 1, E r)
    'rayleigh': (lambda r: 2 * r * np.exp(-r ** 2), np.sqrt(np.pi) / 2),
}
_s = 1.0   # log-normal amplitude, sigma of log r, normalised so that E r^2 = 1
_mu = -_s ** 2
AMPLITUDES['lognormal'] = (lambda r: np.exp(-(np.log(r) - _mu) ** 2 / (2 * _s ** 2)) / (r * _s * np.sqrt(2 * np.pi)) if r > 0 else 0.0,
                           np.exp(_mu + _s ** 2 / 2))


def expected(d, gam, dist='rayleigh'):
    """E J for the nonlinear optimum and for the best amplitude-linear law (E r^2 = 1)."""
    from scipy.integrate import quad
    Vfun = make_Vfun(d)
    f, Er = AMPLITUDES[dist]
    nl = quad(lambda r: r ** 2 * Vfun(gam / r) * f(r) if r > 0 else 0.0, 0, 60, limit=800, points=[0.5, 1, 2, 5])[0]
    lin = Vfun(gam * Er)
    return nl, lin
