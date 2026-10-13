"""
Quasi-linear (mean-flow distortion) model: a single Fourier mode (kx, kz) of the eddy-viscosity
Orr--Sommerfeld--Squire model interacting with the distortion U1(y, t) of the mean profile.

    fluctuation:   dq/dt = A0 q + N(U1) q + B phi           (OSS operator with U -> U + U1)
    mean flow:     dU1/dt = d/dy(nu_T dU1/dy) - d/dy <u'v'>,  U1(+-1) = 0,  <u'v'> = 2 Re(u v*)

The fluctuation is written in the energy coordinates z of lq.Plant (E = |z|^2), U1 as its values
at the interior Gauss--Lobatto points.  Time integration: exponential time differencing (ETD-RK2)
with the stiff linear parts treated exactly; zero-order hold for the control.
Gradients of the cost with respect to the control history by automatic differentiation (JAX).
"""
import numpy as np
import scipy.linalg as sla
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def phi_matrices(A, h):
    """exp(A h), h*phi1(A h), h*phi2(A h) via an augmented matrix exponential."""
    n = A.shape[0]
    Z = np.zeros((3 * n, 3 * n), complex)
    Z[:n, :n] = A * h
    Z[:n, n:2 * n] = np.eye(n) * h
    Z[n:2 * n, 2 * n:] = np.eye(n) * h
    E = sla.expm(Z)
    # E[:n, n:2n] = h phi1(Ah) * h ... careful with scaling: expm([[Ah, hI, 0],[0,0,hI],[0,0,0]])
    E0 = E[:n, :n]
    P1 = E[:n, n:2 * n]            # = h * phi1(A h)
    P2 = E[:n, 2 * n:] / h         # = h * phi2(A h)
    return E0, P1, P2


class QLModel:
    def __init__(self, pl):
        mod = pl.mod
        self.pl = pl
        nos = mod.nos
        self.nos = nos
        a, b, k2 = mod.alp, mod.bet, mod.k2
        D0, D1, D2 = mod.D0, mod.D1, mod.D2
        iD0 = np.linalg.inv(D0)
        self.Dn1 = D1 @ iD0; self.Dn2 = D2 @ iD0               # nodal differentiation
        Linv = np.linalg.inv(pl.L.T)                            # q = Linv z
        self.Lt, self.Linv = pl.L.T, Linv
        # nodal values from z:  v, dv, d2v, eta, deta (interior and walls)
        Zq = np.zeros((nos, nos))
        self.Vn = np.hstack([D0, Zq]) @ Linv
        self.dVn = np.hstack([D1, Zq]) @ Linv
        self.d2Vn = np.hstack([D2, Zq]) @ Linv
        self.En = np.hstack([Zq, D0]) @ Linv
        self.dEn = np.hstack([Zq, D1]) @ Linv
        self.a, self.b, self.k2 = a, b, k2
        # OS left-hand side (with boundary rows) and Squire left-hand side, as in turchan
        B11 = mod.B11
        iB11 = np.linalg.inv(B11)
        rows = np.zeros(nos); rows[2:nos - 2] = 1.0            # interior OS rows
        rowsq = np.zeros(nos); rowsq[1:nos - 1] = 1.0          # interior Squire rows
        # N(U1) q in z coordinates:  -1j * L^T [ iB11 (a U1 (D2-k2 D0) v - a U1'' v) ; iD0 (b U1' v + a U1 eta) ]
        self.iB11, self.iD0, self.rows, self.rowsq = iB11, iD0, rows, rowsq
        self.lap = (D2 - k2 * D0)
        self.D0c = D0
        # mean-flow operator at interior nodes, Dirichlet U1(+-1) = 0
        nut = mod.nut[:, 0]; nutp = mod.nut[:, 1]
        Lm = nut[:, None] * self.Dn2 + nutp[:, None] * self.Dn1
        self.Lm = Lm[1:-1, 1:-1]
        self.nint = nos - 2
        # quadrature weights for the mean-flow energy int U1^2 dy (Clenshaw-Curtis on the nodes)
        from drag import cc_weights
        _, wcc = cc_weights(nos - 1)
        self.wcc = wcc
        self.A0, self.B = pl.A, pl.B
        self.y = mod.y

    # --- nonlinear terms (JAX) ---------------------------------------------------------
    def build(self, h):
        self.h = h
        self.Eq, self.P1q, self.P2q = [jnp.asarray(M) for M in phi_matrices(self.A0, h)]
        self.Em, self.P1m, self.P2m = [jnp.asarray(M.real) for M in phi_matrices(self.Lm.astype(complex), h)]
        J = lambda M: jnp.asarray(M)
        self.jVn, self.jdVn, self.jd2Vn, self.jEn, self.jdEn = J(self.Vn), J(self.dVn), J(self.d2Vn), J(self.En), J(self.dEn)
        self.jLap = J(self.lap @ self.Linv[:self.nos])            # (D2 - k2 D0) v coefficients -> nodes
        self.jD0v = J(self.D0c @ self.Linv[:self.nos])
        self.jiB11, self.jiD0, self.jLt = J(self.iB11), J(self.iD0), J(self.Lt)
        self.jrows, self.jrowsq = J(self.rows), J(self.rowsq)
        self.jDn1, self.jDn2 = J(self.Dn1), J(self.Dn2)
        self.jB = J(self.B)
        self.jwcc = J(self.wcc)

    def U1full(self, U1i):
        return jnp.concatenate([jnp.zeros(1), U1i, jnp.zeros(1)])

    def Nq(self, U1i, z):
        """Perturbation of the OSS operator by the mean-flow distortion, in z coordinates."""
        U1 = self.U1full(U1i)
        U1p = self.jDn1 @ U1
        U1pp = self.jDn2 @ U1
        v = self.jVn @ z; lapv = self.jLap @ z; eta = self.jEn @ z
        os = self.jrows * (self.a * U1 * lapv - self.a * U1pp * v)
        sq = self.jrowsq * (self.b * U1p * v + self.a * U1 * eta)
        qv = self.jiB11 @ os; qe = self.jiD0 @ sq
        return -1j * (self.jLt @ jnp.concatenate([qv, qe]))

    def forcing_mean(self, z):
        """-d/dy <u'v'> at the interior nodes, <u'v'> = 2 Re(u v*)."""
        v = self.jVn @ z; dv = self.jdVn @ z; eta = self.jEn @ z
        u = 1j * (self.a * dv - self.b * eta) / self.k2
        R = 2 * jnp.real(u * jnp.conj(v))
        return -(self.jDn1 @ R)[1:-1]

    def step(self, state, phi):
        """ETD-RK2 (Cox & Matthews) step with zero-order-hold control phi."""
        z, U1 = state
        Fz = self.Nq(U1, z) + self.jB @ phi
        FU = self.forcing_mean(z)
        za = self.Eq @ z + self.P1q @ Fz
        Ua = self.Em @ U1 + self.P1m @ FU
        Fz2 = self.Nq(Ua, za) + self.jB @ phi
        FU2 = self.forcing_mean(za)
        zn = za + self.P2q @ (Fz2 - Fz)
        Un = Ua + self.P2m @ (FU2 - FU)
        return (zn, Un)

    def energies(self, state):
        z, U1 = state
        Ef = jnp.real(jnp.vdot(z, z))
        Em = jnp.sum(self.jwcc * self.U1full(U1) ** 2)
        return Ef, Em

    def simulate(self, z0, U, rho, feedback=None, linear=False):
        """Cost int (E_fluct + rho |phi|^2) dt (trapezoid) for an open-loop control history U (Nt, m)
        or for a feedback law phi_k = feedback(k, z_k).  Returns cost, fluctuation and mean energies."""
        Nt = U.shape[0] if feedback is None else feedback[1]
        h = self.h

        def body(carry, k_u):
            state, J = carry
            k, u = k_u
            if feedback is not None:
                Kk = feedback[0] if feedback[0].ndim == 2 else feedback[0][k]   # (m, n) or (Nt, m, n)
                u = -(Kk @ state[0])
            Ef, Em = self.energies(state)
            wk = jnp.where(k == 0, 0.5, 1.0)
            J = J + h * wk * Ef + rho * h * jnp.real(jnp.vdot(u, u))
            if linear:
                z, U1 = state
                state = (self.Eq @ z + self.P1q @ (self.jB @ u), U1)
            else:
                state = self.step(state, u)
            return (state, J), (Ef, Em)

        state0 = (jnp.asarray(z0), jnp.zeros(self.nint))
        ks = jnp.arange(Nt)
        Us = jnp.asarray(U) if feedback is None else jnp.zeros((Nt, self.B.shape[1]), complex)
        (stateT, J), (Efs, Ems) = jax.lax.scan(body, (state0, 0.0), (ks, Us))
        EfT, EmT = self.energies(stateT)
        J = J + 0.5 * h * EfT
        return J, jnp.append(Efs, EfT), jnp.append(Ems, EmT)
