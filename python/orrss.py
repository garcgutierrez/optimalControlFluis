"""
Orr--Sommerfeld--Squire operator with a Cess eddy viscosity for the turbulent
channel (del Alamo & Jimenez 2006), ported from the MATLAB codes in
../matlab (Dmat.m, turprof_generic.m, turchan.m, prenorm.m,
optimalu2.m, OrrEq.m).

State vector: Chebyshev coefficients q = [v_0..v_N, eta_0..eta_N] of the
wall-normal velocity and wall-normal vorticity of a single Fourier mode
exp(i(alp x + bet z)).  The evolution is dq/dt = -1j * d q = A q.
Boundary conditions (v = Dv = eta = 0) are imposed as relaxation rows
d/dt (BC) = -R_bc (BC) exactly as in turchan.m ("er" trick).

Two scalings are available:
  units='bulk'  : lengths with h, velocities with the bulk velocity U_b
                  (identical to the MATLAB code, used for validation)
  units='tau'   : lengths with h, velocities with u_tau  (used in the paper)
"""
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize_scalar

KAPPA = 0.426
AINT = 25.4


# ----------------------------------------------------------------------------
# Chebyshev machinery (Dmat.m, prenorm.m)
# ----------------------------------------------------------------------------
def dmat(N):
    """Values of T_j and of its first four derivatives at y_i = cos(pi i/N)."""
    vec = np.arange(N + 1)
    y = np.cos(np.pi * vec / N)
    D0 = np.cos(np.outer(vec, vec) * np.pi / N)
    lv = N + 1
    D1 = np.zeros((lv, lv)); D2 = np.zeros((lv, lv))
    D3 = np.zeros((lv, lv)); D4 = np.zeros((lv, lv))
    D1[:, 1] = D0[:, 0]
    D1[:, 2] = 4 * D0[:, 1]
    D2[:, 2] = 4 * D0[:, 0]
    for j in range(3, N + 1):
        D1[:, j] = 2 * j * D0[:, j - 1] + j * D1[:, j - 2] / (j - 2)
        D2[:, j] = 2 * j * D1[:, j - 1] + j * D2[:, j - 2] / (j - 2)
        D3[:, j] = 2 * j * D2[:, j - 1] + j * D3[:, j - 2] / (j - 2)
        D4[:, j] = 2 * j * D3[:, j - 1] + j * D4[:, j - 2] / (j - 2)
    return D0, D1, D2, D3, D4, y


def twonorm(nos):
    """c_ij = int_{-1}^{1} T_i T_j dy."""
    c = np.zeros((nos, nos))
    i = np.arange(nos)[:, None]
    j = np.arange(nos)[None, :]
    even = ((i + j) % 2) == 0
    with np.errstate(divide='ignore'):
        val = 1.0 / (1.0 - (i + j) ** 2) + 1.0 / (1.0 - (i - j) ** 2)
    c[even] = val[even]
    return c


def weinorm(nos, wei):
    """Weighted Chebyshev norm, port of prenorm.m/weinorm (validation only)."""
    from scipy.interpolate import CubicSpline
    NN = max(1024, 2 * nos) + 1
    N2 = 2 * (NN - 1)
    dt = np.pi / N2
    t = np.arange(NN) * dt
    yy = 1 - np.cos(t)
    wei = wei[wei[:, 0] <= 1, :]
    order = np.argsort(wei[:, 0])
    cs = CubicSpline(wei[order, 0], wei[order, 1], extrapolate=True)
    w = cs(yy) * dt
    tab = np.zeros(2 * nos)
    for i in range(1, 2 * nos):
        s = np.sin(i * t) * w
        tab[i] = np.sum(s[0:NN - 2:2] + 4 * s[1:NN - 1:2] + s[2:NN:2]) * 0.5 / 3
    c = np.zeros((nos, nos))
    for i in range(nos):
        for j in range(i % 2, i + 1, 2):
            i1 = 1 + i + j; i2 = 1 - i - j; i3 = 1 + i - j; i4 = 1 - i + j
            val = (tab[abs(i1)] * np.sign(i1) + tab[abs(i2)] * np.sign(i2)
                   + tab[abs(i3)] * np.sign(i3) + tab[abs(i4)] * np.sign(i4))
            c[i, j] = val
            c[j, i] = val
    return c


def coef_derivative(nos):
    """Matrix D such that D a are the Chebyshev coefficients of the derivative."""
    D = np.zeros((nos, nos))
    D[0, 1:nos:2] = np.arange(1, nos, 2)
    for i in range(2, nos + 1):          # MATLAB row index i
        cols = np.arange(i + 1, nos + 1, 2)  # MATLAB column indices
        D[i - 1, cols - 1] = 2 * (cols - 1)
    return D


# ----------------------------------------------------------------------------
# Mean profile and eddy viscosity
# ----------------------------------------------------------------------------
def cess_nut_plus(eta, Rt, kapa=KAPPA, Aint=AINT):
    """Total (molecular + eddy) viscosity nu_T/nu from Cess (1958)."""
    return 0.5 * np.sqrt(1 + kapa ** 2 * Rt ** 2 / 9 * (2 * eta - eta ** 2) ** 2
                         * (3 - 4 * eta + 2 * eta ** 2) ** 2
                         * (1 - np.exp(-eta * Rt / Aint)) ** 2) + 0.5


def turprof_generic(N, Aint, kapa, eddyfrac, Rt):
    """Verbatim port of turprof_generic.m (bulk units, finite-difference derivatives)."""
    y = np.cos(np.pi * np.arange(N + 1) / N)
    Nos = N + 1
    y0 = 1 - y
    nut = cess_nut_plus(y0, Rt, kapa, Aint)
    f = (1 - y0) / nut
    u = Rt * np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(y0))])
    u = u - u[0]
    du = -y / nut
    n0 = Nos // 2
    n1 = Nos - n0
    du = np.concatenate([du[:n1], -du[n0 - 1::-1]])
    nut = np.concatenate([nut[:n1], nut[n0 - 1::-1]])
    u = np.concatenate([u[:n1], u[n0 - 1::-1]])
    Ub = np.trapezoid(u, y0) / np.trapezoid(np.ones(Nos), y0)
    u = u / Ub
    Re = Ub * Rt
    nut = nut / Re
    du = du * Rt / Ub

    def fd(a, sgn_end):
        ap = np.zeros_like(a)
        ap[1:-1] = (a[2:] - a[:-2]) / (y[2:] - y[:-2])
        ap[0] = (a[1] - a[0]) / (y[1] - y[0])
        ap[-1] = sgn_end * ap[0]
        return ap

    upp = fd(du, +1)
    nut = nut[0] + (nut - nut[0]) * eddyfrac
    nutp = fd(nut, -1)
    nutpp = fd(nutp, +1)
    nutppp = fd(nutpp, -1)
    return y, np.column_stack([nut, nutp, nutpp, nutppp]), np.column_stack([u, du, upp]), Ub


def turprof_spectral(N, Rt, eddyfrac=1.0, units='tau', kapa=KAPPA, Aint=AINT, Nfine=4096):
    """Accurate mean profile and eddy viscosity (and derivatives) on the
    Chebyshev nodes.  U is obtained by integrating dU+/dy+ = (1-eta)/nu_T+
    with high-order quadrature, derivatives of nu_T are computed spectrally."""
    y = np.cos(np.pi * np.arange(N + 1) / N)
    # fine Chebyshev representation of nu_T(y) over the whole channel
    Mf = 512
    yf = np.cos(np.pi * np.arange(Mf + 1) / Mf)
    nut_f = cess_nut_plus(1 - np.abs(yf), Rt, kapa, Aint)
    nut_f = nut_f[0] + (nut_f - nut_f[0]) * eddyfrac  # damping as in the MATLAB code
    cheb = np.polynomial.chebyshev.Chebyshev.fit(yf, nut_f, Mf, domain=[-1, 1])
    # the fit is an interpolant on the extrema grid; derivatives spectrally
    nut = cheb(y); nutp = cheb.deriv(1)(y); nutpp = cheb.deriv(2)(y); nutppp = cheb.deriv(3)(y)
    # mean velocity: U+(eta) = Re_tau int_0^eta (1-s)/nu_T+(s) ds  (full eddy viscosity)
    from scipy.integrate import quad
    def dUdeta(s):
        return Rt * (1 - s) / cess_nut_plus(s, Rt, kapa, Aint)
    eta = 1 - np.abs(y)
    U = np.array([quad(dUdeta, 0.0, e, limit=400, epsabs=1e-12, epsrel=1e-12)[0] for e in eta])
    dU = -np.sign(y) * dUdeta(eta)          # dU/dy
    # d2U/dy2 analytic: d/deta[(1-s)/nu] = -1/nu - (1-s) nu'/nu^2 ; dy = -sign(y) deta
    s = eta
    nu = cess_nut_plus(s, Rt, kapa, Aint)
    h = 1e-6
    dnu = (cess_nut_plus(s + h, Rt, kapa, Aint) - cess_nut_plus(np.maximum(s - h, 0), Rt, kapa, Aint)) \
        / (s + h - np.maximum(s - h, 0))
    upp = Rt * (-1.0 / nu - (1 - s) * dnu / nu ** 2)   # sign(y)^2 = 1
    # bulk velocity (plus units) with Clenshaw-Curtis on a fine grid
    ee = np.linspace(0, 1, 20001)
    from scipy.integrate import cumulative_trapezoid
    Uf = cumulative_trapezoid(dUdeta(ee), ee, initial=0.0)
    Ubp = np.trapezoid(Uf, ee)
    if units == 'tau':
        scaleU, scaleNu = 1.0, 1.0 / Rt           # U in u_tau, nu in u_tau h
    elif units == 'bulk':
        scaleU, scaleNu = 1.0 / Ubp, 1.0 / (Rt * Ubp)
    else:
        raise ValueError(units)
    nutc = np.column_stack([nut, nutp, nutpp, nutppp]) * scaleNu
    uc = np.column_stack([U, dU, upp]) * scaleU
    return y, nutc, uc, Ubp


# ----------------------------------------------------------------------------
# Operator (turchan.m)
# ----------------------------------------------------------------------------
def turchan(alp, bet, D0, D1, D2, D3, D4, uc, nut, Rbc):
    """Returns d (dq/dt = -1j d q), and the collocation matrices B11 (OS lhs)
    so that forcing can be added consistently."""
    zi = 1j
    ak2 = alp ** 2 + bet ** 2
    Nos = uc.shape[0]
    u, du, upp = uc[:, 0], uc[:, 1], uc[:, 2]
    nt, ntp, ntpp, ntppp = nut[:, 0], nut[:, 1], nut[:, 2], nut[:, 3]
    er = -Rbc * zi
    B11 = D2 - ak2 * D0
    A11 = alp * u[:, None] * B11
    A11 = A11 - alp * upp[:, None] * D0
    A11 = A11 + zi * nt[:, None] * (D4 - 2 * ak2 * D2 + ak2 ** 2 * D0)
    A11 = A11 + zi * ntp[:, None] * (3 * D3 - 3 * ak2 * D1)
    A11 = A11 + zi * ntpp[:, None] * (3 * D2 - ak2 * D0)
    A11 = A11 + zi * ntppp[:, None] * D1
    A11 = np.vstack([er * D0[0], er * D1[0], A11[2:Nos - 2], er * D1[-1], er * D0[-1]])
    B11 = np.vstack([D0[0], D1[0], B11[2:Nos - 2], D1[-1], D0[-1]])
    A21 = bet * du[:, None] * D0
    A21 = np.vstack([np.zeros(Nos), A21[1:Nos - 1], np.zeros(Nos)])
    A22 = -(nt[:, None] * (D2 - ak2 * D0)) / zi
    A22 = A22 - (ntp[:, None] * D1) / zi
    A22 = A22 + alp * u[:, None] * D0
    A22 = np.vstack([er * D0[0], A22[1:Nos - 1], er * D0[-1]])
    top = np.hstack([np.linalg.solve(B11, A11), np.zeros((Nos, Nos), complex)])
    bot = np.linalg.solve(D0, np.hstack([A21, A22]))
    d = np.vstack([top, bot])
    return d, B11


def energy_matrix(C, D, alp, bet):
    """q^H Q q = int (|v|^2 + |Dv|^2/k^2 + |eta|^2/k^2) dy  (calcEner in OrrEq.m)."""
    ak2 = 1.0 / (alp ** 2 + bet ** 2)
    nos = C.shape[0]
    Q = np.zeros((2 * nos, 2 * nos))
    Q[:nos, :nos] = D.T @ C @ D * ak2 + C
    Q[nos:, nos:] = ak2 * C
    return Q


def bc_matrix(D0, D1):
    """Rows giving v(1), Dv(1), Dv(-1), v(-1), eta(1), eta(-1) from q."""
    nos = D0.shape[0]
    Z = np.zeros(nos)
    rows = [np.r_[D0[0], Z], np.r_[D1[0], Z], np.r_[D1[-1], Z], np.r_[D0[-1], Z],
            np.r_[Z, D0[0]], np.r_[Z, D0[-1]]]
    return np.array(rows)


def herm(M):
    return 0.5 * (M + M.conj().T)


class OrrModel:
    """Container for one wavenumber pair."""

    def __init__(self, alp, bet, N=128, Rt=2000.0, eddyfrac=1.0, units='tau', legacy=False,
                 Rbc=None, profile=None):
        self.alp, self.bet, self.N, self.Rt = alp, bet, N, Rt
        self.k2 = alp ** 2 + bet ** 2
        D0, D1, D2, D3, D4, y = dmat(N)
        self.D0, self.D1, self.D2, self.D3, self.y = D0, D1, D2, D3, y
        if profile is not None:
            y_, nut, uc, Ubp = profile
        elif legacy:
            y_, nut, uc, Ubp = turprof_generic(N, AINT, KAPPA, eddyfrac, Rt)
            if units == 'tau':
                raise ValueError('legacy profile only in bulk units')
        else:
            y_, nut, uc, Ubp = turprof_spectral(N, Rt, eddyfrac, units)
        self.nut, self.uc, self.Ubp = nut, uc, Ubp
        # relaxation rate of the boundary rows: Rt in bulk units as in turchan.m
        if Rbc is None:
            Rbc = Rt if units == 'bulk' else Rt * Ubp
        self.Rbc = Rbc
        d, B11 = turchan(alp, bet, D0, D1, D2, D3, D4, uc, nut, Rbc)
        self.d, self.B11 = d, B11
        self.A = -1j * d
        nos = N + 1
        self.nos = nos
        self.C = twonorm(nos)
        self.Dc = coef_derivative(nos)
        self.Q = energy_matrix(self.C, self.Dc, alp, bet)
        self.BC = bc_matrix(D0, D1)

    # -- physical subspace (homogeneous BCs), energy-orthonormal basis ------
    def physical_basis(self):
        """Phi (n x m) with Phi^H Q Phi = I spanning {q : BC q = 0}."""
        Z = sla.null_space(self.BC)
        G = herm(Z.conj().T @ self.Q @ Z)
        L = np.linalg.cholesky(G)
        Phi = Z @ np.linalg.inv(L).conj().T
        return Phi

    def reduced(self):
        Phi = self.physical_basis()
        Ar = Phi.conj().T @ self.Q @ self.A @ Phi
        return Ar, Phi

    # -- wall-normal velocity profile of a state -----------------------------
    def velocities(self, q):
        """Physical-space u, v, w at the collocation nodes."""
        nos = self.nos
        v = self.D0 @ q[:nos]
        dv = self.D1 @ q[:nos]
        eta = self.D0 @ q[nos:]
        u = 1j * (self.alp * dv - self.bet * eta) / self.k2
        w = 1j * (self.bet * dv + self.alp * eta) / self.k2
        return u, v, w

    # -- actuators ----------------------------------------------------------
    def B_wall(self):
        """Blowing/suction v(+1)=g1, v(-1)=g2 through the relaxation BC rows."""
        nos = self.nos
        F = np.zeros((nos, 2), complex)
        F[0, 0] = self.Rbc          # row of v(+1)
        F[nos - 1, 1] = self.Rbc    # row of v(-1)
        Bv = np.linalg.solve(self.B11, F)
        return np.vstack([Bv, np.zeros((nos, 2), complex)])

    def B_legacy(self, interv=16):
        """Actuator used in OrrEq.m (validation only)."""
        nos = self.nos
        bbb = np.zeros(nos); bbb1 = np.zeros(nos)
        bbb[:interv] = np.sin(np.arange(interv, 2 * interv) / (2 * interv) * np.pi)
        bbb1[nos - interv:] = np.sin(np.arange(2 * interv - 1, interv - 1, -1) / (2 * interv) * np.pi)
        c1 = np.linalg.solve(self.D0, bbb); c2 = np.linalg.solve(self.D0, bbb1)
        return np.vstack([np.column_stack([c1, c2]), np.zeros((nos, 2))]).astype(complex)


# ----------------------------------------------------------------------------
# optimalu2.m port (validation of the port against the MATLAB data)
# ----------------------------------------------------------------------------
def eigenjj(d, minimag, maxreal, nmin=65, nmax=128):
    e, v = np.linalg.eig(d)
    zer = -1e-13
    ii = np.where((e.real >= zer) & (e.real <= maxreal))[0]
    e = e[ii]; v = v[:, ii]
    idx = np.argsort(-e.imag, kind='stable')
    e = e[idx]; v = v[:, idx]
    ei = e.imag
    ncols = len(e)
    jj = np.where(ei <= minimag)[0]
    nei = jj[0] + 1 if len(jj) else ncols
    keep = min(ncols, min(nmax, max(nmin, nei)))
    return v[:, :keep], e[:keep]


def sqrt_factor(work):
    U, S, Vh = np.linalg.svd(work)
    s = np.sqrt(S)
    return U @ np.diag(1 / s), np.diag(s) @ U.conj().T


def optimalu2_ic(model, Cwei=None, NT=100):
    """Optimal initial condition (iopt=0) following optimalu2.m.
    Returns flowout = xu*kin (Chebyshev coefficients) and tformax."""
    alp, bet = model.alp, model.bet
    ak2 = 1 / (alp ** 2 + bet ** 2)
    xu, eu = eigenjj(model.d, -1, 10 * alp)
    nos = model.nos
    C, D = model.C, model.Dc
    xv = xu[:nos]; xw = xu[nos:]
    work = xv.conj().T @ (D.T @ C @ D * ak2 + C) @ xv + ak2 * xw.conj().T @ C @ xw
    invF, F = sqrt_factor(work)
    if Cwei is None:
        Cwei = C
    work = xv.conj().T @ (D.T @ Cwei @ D * ak2 + Cwei) @ xv + ak2 * xw.conj().T @ Cwei @ xw
    _, Fwei = sqrt_factor(work)
    F2 = Fwei
    T = 40

    def maxer(t):
        return -np.linalg.norm(F2 @ np.diag(np.exp(-1j * t * eu)) @ invF, 2)

    NTT = NT // 2
    tid = np.linspace(0, T, NTT)
    gg = np.array([maxer(t) ** 2 for t in tid])
    ii = np.where(gg >= min(gg[0] / 20, gg[2]))[0]
    gg = gg[ii]
    dgg = np.diff(gg)
    igg = np.where(dgg >= 0)[0]
    if len(igg):
        tf = tid[igg[-1] + 1] * 2; ts = tf / 4
    else:
        ts = 0.0; tf = tid[1]
    res = minimize_scalar(maxer, bounds=(ts, tf), method='bounded', options={'xatol': 1e-5})
    tformax, kg = res.x, res.fun
    if maxer(ts) < kg:
        tformax = ts
    evol = F2 @ np.diag(np.exp(-1j * tformax * eu)) @ invF
    U, S, Vh = np.linalg.svd(evol)
    kin = invF @ Vh.conj().T[:, 0]
    return xu @ kin, tformax
