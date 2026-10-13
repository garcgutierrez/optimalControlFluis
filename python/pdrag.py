"""
Drag objective with an amplitude-proportional actuation cost, WITHOUT full information:
the controller only measures the wall pressure, with noise.

Burst: z(0) = beta z0, beta = b a, b ~ Bernoulli(p), a ~ CN(0,1).  Zero-order hold on Nt intervals.
All the information about beta is in the matched-filter statistic m(t_k) ~ CN(beta h_k, h_k),
with independent increments.  phi_k uses the information up to t_k.

Everything is formulated exactly in the space of the control sequence u (Nt*m complex values):
    q_k = beta x_k + T_k u,   x_k = E^k z0,   T_k = [E^{k-1}G, ..., G, 0, ...]
    J(beta, u) = |beta|^2 D0 + 2 Re(beta u^H b0) + u^H H u + gamma h |u|_1
with H = sum_k w_k T_k^H M T_k + rho h I and b0 = sum_k w_k T_k^H M x_k.

Policies
  best linear output feedback  u_k = sum_{l<=k} G[k,l] dm_l      (exact expected cost, convex)
  thresholded certainty equivalence  u_k = soft(-K_k qhat_k, tau)   (Monte Carlo)
  certainty-equivalent MPC (open-loop feedback): at each step, minimise the expected remaining
     cost given the conditional mean of the state, apply the first control     (Monte Carlo)
  full information: nonlinear optimum E|a|^2 V(gamma/|a|) and best linear V(gamma E|a|).
"""
import numpy as np
from scipy.optimize import minimize

SQPI2 = np.sqrt(np.pi) / 2


def csoft(U, tau, axis_m=-1):
    """Complex soft thresholding of every entry (one actuator per entry)."""
    a = np.abs(U)
    return U * np.maximum(0.0, 1.0 - tau / np.maximum(a, 1e-300))


class InputSpace:
    def __init__(self, disc, M, rho, z0):
        Nt, n, m, h, w = disc.Nt, disc.n, disc.m, disc.h, disc.w
        E, Gm = disc.Eh, disc.Gh
        self.Nt, self.n, self.m, self.h, self.w, self.rho = Nt, n, m, h, w, rho
        Nu = Nt * m
        self.Nu = Nu
        # T[k] (n x Nu): response at t_k to the control sequence
        T = np.zeros((Nt + 1, n, Nu), complex)
        for k in range(1, Nt + 1):
            T[k] = E @ T[k - 1]
            T[k][:, (k - 1) * m:k * m] = Gm
        X = np.zeros((Nt + 1, n), complex); X[0] = z0
        for k in range(Nt):
            X[k + 1] = E @ X[k]
        self.T, self.X = T, X
        MT = np.einsum('ij,kjl->kil', M, T)
        self.H = np.einsum('k,kil,kij->lj', w, T.conj(), MT) + rho * h * np.eye(Nu)
        self.H = 0.5 * (self.H + self.H.conj().T)
        self.b0 = np.einsum('k,kil,ki->l', w, T.conj(), np.einsum('ij,kj->ki', M, X))
        self.D0 = float(np.real(np.einsum('k,ki,ki->', w, X.conj(), np.einsum('ij,kj->ki', M, X))))
        ev = np.linalg.eigvalsh(self.H)
        self.Lmax, self.Lmin = ev.max(), ev.min()
        # data for certainty-equivalent MPC: free response of an arbitrary state q at t_k
        #   b_k(q) = sum_{k'>=k} w_k' T_k'[:, tail]^H M E^{k'-k} q  (tail = inputs k..Nt-1)
        self._E, self._M = E, M

    def cost(self, beta, U):
        """Per-sample cost. beta: (S,), U: (Nu, S)."""
        HU = self.H @ U
        return (np.abs(beta) ** 2 * self.D0 + 2 * np.real(beta * (U.conj() * self.b0[:, None]).sum(0))
                + np.real((U.conj() * HU).sum(0)))

    def l1(self, U):
        return self.h * np.abs(U).sum(0)

    # ---------------- full information --------------------------------------------------
    def value(self, g, U0=None, maxiter=20000, tol=1e-12):
        """V(g) = min_u D0 + 2Re(u^H b0) + u^H H u + g h |u|_1 (unit burst), FISTA."""
        L = self.Lmax * 1.02
        X = np.zeros(self.Nu, complex) if U0 is None else U0.copy()
        Y = X.copy(); tk = 1.0; Jold = np.inf
        f = lambda u: self.D0 + 2 * np.real(u.conj() @ self.b0) + np.real(u.conj() @ self.H @ u) + g * self.h * np.abs(u).sum()
        for it in range(maxiter):
            grad = self.H @ Y + self.b0
            Xn = csoft(Y - grad / L, g * self.h / (2 * L))
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk ** 2))
            Y = Xn + (tk - 1) / tn * (Xn - X); X, tk = Xn, tn
            if it % 20 == 0:
                J = f(X)
                if J > Jold:
                    Y = X.copy(); tk = 1.0
                if abs(Jold - J) < tol * abs(J):
                    break
                Jold = J
        return f(X), X

    def threshold(self):
        return 2 * np.max(np.abs(self.b0)) / self.h


class LinearPolicy:
    """Best linear causal output feedback u_k = sum_{l=1}^{k} G[k,l] dm_l (exact expected cost)."""

    def __init__(self, ins, dh, p, gamma):
        self.I, self.dh, self.p, self.gamma = ins, dh, p, gamma
        Nt, m = ins.Nt, ins.m
        self.mask = np.tril(np.ones((Nt, Nt), bool), 0); self.mask[:, 0] = False
        self.nvar = int(self.mask.sum()) * m

    def unpack(self, x):
        Nt, m = self.I.Nt, self.I.m
        G = np.zeros((Nt, Nt, m), complex)
        G[self.mask] = (x[:self.nvar] + 1j * x[self.nvar:]).reshape(-1, m)
        return G

    def cost_grad(self, x, eps):
        I, dh, p, gam = self.I, self.dh, self.p, self.gamma
        Nt, m = I.Nt, I.m
        G = self.unpack(x)
        c = np.einsum('klj,l->kj', G, dh)
        s2 = np.einsum('klj,l->kj', np.abs(G) ** 2, dh)
        cv = c.reshape(-1)
        Hc = I.H @ cv
        J = p * (I.D0 + 2 * np.real(cv.conj() @ I.b0) + np.real(cv.conj() @ Hc))
        gc = p * (Hc + I.b0).reshape(Nt, m)
        Gm = G.transpose(0, 2, 1).reshape(Nt * m, Nt)           # column l = input sequence for dm_l
        HG = I.H @ Gm
        J += float(np.real(np.sum(Gm.conj() * HG, 0)) @ dh)
        gG = (HG * dh[None, :]).reshape(Nt, m, Nt).transpose(0, 2, 1)
        if gam > 0:
            r1 = np.sqrt(np.abs(c) ** 2 + s2 + eps ** 2); r0 = np.sqrt(s2 + eps ** 2)
            J += gam * I.h * SQPI2 * float(np.sum(p * (r1 - eps) + (1 - p) * (r0 - eps)))
            gc = gc + gam * I.h * SQPI2 * p * c / (2 * r1)
            fac = gam * I.h * SQPI2 * (p / (2 * r1) + (1 - p) / (2 * r0))
            gG = gG + fac[:, None, :] * G * dh[None, :, None]
        gG = gG + gc[:, None, :] * dh[None, :, None]
        gG[~self.mask] = 0
        v = gG[self.mask].ravel()
        return J, np.concatenate([2 * v.real, 2 * v.imag])

    def exact_cost(self, x):
        return self.cost_grad(x, 1e-300)[0]

    def solve(self, scale, x0=None, eps_list=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), maxiter=5000):
        """Smoothed l1 with continuation in eps (relative to the typical control amplitude)."""
        x = np.zeros(2 * self.nvar) if x0 is None else x0
        info = []
        for e in eps_list:
            res = minimize(lambda z: self.cost_grad(z, e * scale), x, jac=True, method='L-BFGS-B',
                           options=dict(maxiter=maxiter, maxcor=50, ftol=1e-15, gtol=1e-12))
            x = res.x
            info.append((e, res.nit, self.exact_cost(x)))
        return x, self.exact_cost(x), info


class MonteCarlo:
    def __init__(self, ins, hgrid, p, S=4000, seed=0):
        self.I, self.h, self.p, self.S = ins, hgrid, p, S
        rng = np.random.default_rng(seed)
        Nt = ins.Nt
        cn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)
        self.b = (rng.random(S) < p).astype(float)
        self.a = cn(S)
        self.beta = self.b * self.a
        dh = np.diff(hgrid)
        dm = self.beta[None, :] * dh[:, None] + np.sqrt(dh)[:, None] * cn(Nt, S)
        self.m = np.vstack([np.zeros(S), np.cumsum(dm, 0)])

    def estimate(self, k, kind):
        h, m, p = self.h[k], self.m[k], self.p
        if kind == 'kalman':
            return p * m / (1 + p * h)
        if p >= 1:
            return m / (1 + h)
        q = (1 - p) / p * (1 + h)
        pi = 1.0 / (1.0 + q * np.exp(-np.minimum(np.abs(m) ** 2 / (1 + h), 700)))
        return pi * m / (1 + h)

    def total(self, U, gamma):
        return self.I.cost(self.beta, U) + gamma * self.I.l1(U)

    def run_linear(self, G, gamma):
        I = self.I; Nt, m = I.Nt, I.m
        dm = np.diff(self.m, axis=0)
        U = np.zeros((Nt, m, self.S), complex)
        for k in range(1, Nt):
            U[k] = np.einsum('lj,ls->js', G[k, 1:k + 1, :], dm[:k])
        return self.total(U.reshape(Nt * m, self.S), gamma)

    def run_feedback(self, K, tau, kind, gamma):
        """u_k = soft(-K_k (T_k u + betahat x_k), tau)."""
        I = self.I; Nt, m = I.Nt, I.m
        KT = [K[k] @ I.T[k] for k in range(Nt)]
        Kx = [K[k] @ I.X[k] for k in range(Nt)]
        U = np.zeros((Nt * m, self.S), complex)
        for k in range(Nt):
            bh = self.estimate(k, kind)
            u = -(KT[k] @ U + Kx[k][:, None] * bh[None, :])
            if tau > 0:
                u = csoft(u, tau)
            U[k * m:(k + 1) * m] = u
        return self.total(U, gamma)

    def run_mpc(self, gamma, kind, iters=300):
        """Certainty-equivalent MPC: at step k minimise the remaining cost for the state estimate
        qhat_k = T_k u + betahat x_k (exact expected cost of the remaining open-loop sequence,
        up to a constant) and apply the first control."""
        I = self.I; Nt, m, S, h = I.Nt, I.m, self.S, I.h
        U = np.zeros((Nt * m, S), complex)
        Ufut = np.zeros((Nt * m, S), complex)          # warm start
        for k in range(Nt):
            sl = slice(k * m, Nt * m)
            Hkk = I.H[sl, sl]
            L = np.linalg.eigvalsh(Hkk).max() * 1.02
            bh = self.estimate(k, kind)
            # linear term: cross terms with the past controls and the estimated burst
            bvec = I.H[sl, :k * m] @ U[:k * m] + I.b0[sl][:, None] * bh[None, :]
            X = Ufut[sl].copy(); Y = X.copy(); tk = 1.0
            for it in range(iters):
                grad = Hkk @ Y + bvec
                Xn = csoft(Y - grad / L, gamma * h / (2 * L))
                tn = 0.5 * (1 + np.sqrt(1 + 4 * tk ** 2))
                Y = Xn + (tk - 1) / tn * (Xn - X); X, tk = Xn, tn
            U[k * m:(k + 1) * m] = X[:m]
            Ufut[sl] = X
        return self.total(U, gamma)


def lq_gains(disc, M, rho_d):
    """Finite-horizon discrete LQ gains for the state weight M and penalty rho_d h."""
    n, m, Nt, h = disc.n, disc.m, disc.Nt, disc.h
    E, G, w = disc.Eh, disc.Gh, disc.w
    Sm = w[-1] * M
    K = [None] * Nt
    for k in range(Nt - 1, -1, -1):
        R = rho_d * h * np.eye(m) + G.conj().T @ Sm @ G
        K[k] = np.linalg.solve(R, G.conj().T @ Sm @ E)
        Sm = w[k] * M + E.conj().T @ Sm @ (E - G @ K[k])
        Sm = 0.5 * (Sm + Sm.conj().T)
    return K
