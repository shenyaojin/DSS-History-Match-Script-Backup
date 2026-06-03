"""
Adjoint-identity verification for the one-way coupled poroelastic
permeability inversion described in 2026_Jin_PermInvReport_V1.
 
The report derives, via a Lagrangian / integration-by-parts argument:
  (1) adjoint solid eqns   : K^T u_dag = misfit dipole source
  (2) adjoint fluid eqn    : backward-in-time diffusion sourced by alpha*div(u_dag)
  (3) gradient             : dJ/dk_xx = -int (1/mu) (dp_dag/dx)(dp/dx) dt
 
This script builds a SMALL discrete analogue with the SAME operator chain
  k  -->  D(k) (diffusion)  -->  p(t)  -->  body force -alpha*grad(p)
       -->  K (elasticity)  -->  u(t)  -->  strain_yy at the fiber  =  g(t)
and verifies the adjoint method with two independent, standard tests:
 
  TEST 1 - Adjoint (dot-product) identity:   <L dm, w> == <dm, L* w>
           where L is the tangent-linear operator (dm -> dg) and
           L* is the adjoint operator implemented from the derived equations.
           A correct adjoint passes this to machine precision for random dm, w.
 
  TEST 2 - Taylor / finite-difference gradient test:
           the adjoint gradient g must satisfy
              J(m + e dm) - J(m - e dm) = 2 e <g, dm> + O(e^3),
           i.e. the central-difference directional derivative matches <g,dm>.
 
All comments are in English by request.
"""
 
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
 
rng = np.random.default_rng(0)
 
# ----------------------------------------------------------------------
# Grid and physical constants (values are O(1); units are irrelevant for
# verifying the adjoint identity, which is a purely algebraic property).
# ----------------------------------------------------------------------
Nx, Ny = 9, 9          # nodes in x and y
h      = 1.0           # uniform spacing
Nn     = Nx * Ny       # number of pressure nodes
lam, G = 1.3, 0.9      # Lame parameters
alpha  = 1.0           # Biot coefficient
mu     = 1.0           # fluid viscosity
s      = 1.0           # storage coefficient phi0*cf
Nt     = 16            # time steps
dt     = 0.5
i_fib  = Nx - 2        # fiber (observation) column index in x
 
 
def idx(i, j):
    """Flatten 2D node (i,j) to a scalar pressure-dof index."""
    return i * Ny + j
 
 
# ----------------------------------------------------------------------
# Diffusion operator D(k):  discrete  div( k/mu grad(.) )  with Neumann BC.
# Built in divergence form with arithmetic face permeabilities, so D is
# symmetric. D is LINEAR in the nodal permeability field k, hence we
# precompute the per-node "unit-field" operators Dc with D(k) = sum_c k_c Dc.
# ----------------------------------------------------------------------
def diffusion_operator(kfield):
    rows, cols, vals = [], [], []
 
    def add(a, b, t):
        rows.append(a); cols.append(b); vals.append(t)
 
    for i in range(Nx):
        for j in range(Ny):
            P = idx(i, j)
            # x+ face
            if i + 1 < Nx:
                t = 0.5 * (kfield[P] + kfield[idx(i + 1, j)]) / mu / h**2
                add(P, idx(i + 1, j), t); add(P, P, -t)
            # x- face
            if i - 1 >= 0:
                t = 0.5 * (kfield[P] + kfield[idx(i - 1, j)]) / mu / h**2
                add(P, idx(i - 1, j), t); add(P, P, -t)
            # y+ face
            if j + 1 < Ny:
                t = 0.5 * (kfield[P] + kfield[idx(i, j + 1)]) / mu / h**2
                add(P, idx(i, j + 1), t); add(P, P, -t)
            # y- face
            if j - 1 >= 0:
                t = 0.5 * (kfield[P] + kfield[idx(i, j - 1)]) / mu / h**2
                add(P, idx(i, j - 1), t); add(P, P, -t)
    return sp.csr_matrix((vals, (rows, cols)), shape=(Nn, Nn))
 
 
# Precompute Dc for each node: D(k) = sum_c k_c * Dc  (exact, since D is linear in k).
Dc = [diffusion_operator(np.eye(Nn)[c]) for c in range(Nn)]
 
 
def Dmat(kfield):
    out = Dc[0] * kfield[0]
    for c in range(1, Nn):
        out = out + Dc[c] * kfield[c]
    return out
 
 
# ----------------------------------------------------------------------
# Elasticity operator K (the discrete Navier operator from the report) and
# the poroelastic coupling G_op so that the solid RHS body force = G_op @ p
# (the discrete -alpha*grad(p)). K need not be symmetric for the test -- the
# adjoint solid solve uses K^T explicitly, which is the honest discrete adjoint.
# DOFs: ux at 0..Nn-1, uy at Nn..2Nn-1.
# ----------------------------------------------------------------------
Ndof = 2 * Nn
 
 
def ux(i, j): return idx(i, j)
def uy(i, j): return Nn + idx(i, j)
 
 
def assemble_elasticity():
    rows, cols, vals = [], [], []
 
    def add(a, b, v):
        rows.append(a); cols.append(b); vals.append(v)
 
    inv = 1.0 / h**2
    for i in range(Nx):
        for j in range(Ny):
            # ---- x-momentum: (lam+2G) uxx + G uyy + (lam+G) d2uy/dxdy ----
            r = ux(i, j)
            if 0 < i < Nx - 1:
                add(r, ux(i + 1, j), (lam + 2 * G) * inv)
                add(r, ux(i - 1, j), (lam + 2 * G) * inv)
                add(r, ux(i, j), -2 * (lam + 2 * G) * inv)
            else:
                add(r, ux(i, j), 1.0)  # Neumann-ish edge stabilization
            if 0 < j < Ny - 1:
                add(r, ux(i, j + 1), G * inv)
                add(r, ux(i, j - 1), G * inv)
                add(r, ux(i, j), -2 * G * inv)
            if 0 < i < Nx - 1 and 0 < j < Ny - 1:
                c = (lam + G) / (4 * h**2)
                add(r, uy(i + 1, j + 1), c)
                add(r, uy(i - 1, j - 1), c)
                add(r, uy(i + 1, j - 1), -c)
                add(r, uy(i - 1, j + 1), -c)
 
            # ---- y-momentum: G uxx + (lam+2G) uyy + (lam+G) d2ux/dxdy ----
            r = uy(i, j)
            if 0 < i < Nx - 1:
                add(r, uy(i + 1, j), G * inv)
                add(r, uy(i - 1, j), G * inv)
                add(r, uy(i, j), -2 * G * inv)
            else:
                add(r, uy(i, j), 1.0)
            if 0 < j < Ny - 1:
                add(r, uy(i, j + 1), (lam + 2 * G) * inv)
                add(r, uy(i, j - 1), (lam + 2 * G) * inv)
                add(r, uy(i, j), -2 * (lam + 2 * G) * inv)
            if 0 < i < Nx - 1 and 0 < j < Ny - 1:
                c = (lam + G) / (4 * h**2)
                add(r, ux(i + 1, j + 1), c)
                add(r, ux(i - 1, j - 1), c)
                add(r, ux(i + 1, j - 1), -c)
                add(r, ux(i - 1, j + 1), -c)
 
    K = sp.csr_matrix((vals, (rows, cols)), shape=(Ndof, Ndof)).tolil()
    # Pin a few dofs so K is invertible (removes near-null directions).
    for d in (ux(0, 0), uy(0, 0), ux(Nx - 1, Ny - 1)):
        K.rows[d] = [d]; K.data[d] = [1.0]
    return K.tocsr()
 
 
def assemble_coupling():
    """G_op: pressure space -> displacement space, body force = -alpha*grad(p)."""
    rows, cols, vals = [], [], []
 
    def add(a, b, v):
        rows.append(a); cols.append(b); vals.append(v)
 
    for i in range(Nx):
        for j in range(Ny):
            if 0 < i < Nx - 1:
                add(ux(i, j), idx(i + 1, j), -alpha / (2 * h))
                add(ux(i, j), idx(i - 1, j),  alpha / (2 * h))
            if 0 < j < Ny - 1:
                add(uy(i, j), idx(i, j + 1), -alpha / (2 * h))
                add(uy(i, j), idx(i, j - 1),  alpha / (2 * h))
    return sp.csr_matrix((vals, (rows, cols)), shape=(Ndof, Nn))
 
 
def assemble_observation():
    """E: displacement space -> fiber strain_yy = d(uy)/dy at column i_fib."""
    rows, cols, vals = [], [], []
    obs = []
    for j in range(Ny):
        r = len(obs)
        if 0 < j < Ny - 1:
            rows += [r, r]; cols += [uy(i_fib, j + 1), uy(i_fib, j - 1)]
            vals += [1 / (2 * h), -1 / (2 * h)]
        obs.append((i_fib, j))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(obs), Ndof)), len(obs)
 
 
K_mat = assemble_elasticity()
G_op  = assemble_coupling()
E, Nobs = assemble_observation()
 
K_lu  = spla.splu(sp.csc_matrix(K_mat))
KT_lu = spla.splu(sp.csc_matrix(K_mat.T))
 
# Time-varying volumetric injection source at the producer column (i=0).
q = np.zeros((Nt + 1, Nn))
for n in range(1, Nt + 1):
    amp = np.sin(np.pi * n / Nt)               # smooth recharge-like profile
    for j in range(Ny):
        q[n, idx(0, j)] = amp
 
 
# ----------------------------------------------------------------------
# Forward model:  m = log(k)  ->  p(t), u(t), g(t) = strain_yy at fiber
# Implicit Euler in time for the linear diffusion; static solid solve per step.
#   A p^n = (M/dt) p^{n-1} + q^n ,  A = M/dt - D(k) ,  M = s I
#   K u^n = G_op p^n ;  g^n = E u^n
# ----------------------------------------------------------------------
def forward(m):
    k = np.exp(m)
    D = Dmat(k)
    M_over_dt = (s / dt) * sp.identity(Nn, format="csr")
    A = (M_over_dt - D).tocsc()
    A_lu = spla.splu(A)
 
    p = np.zeros((Nt + 1, Nn))
    u = np.zeros((Nt + 1, Ndof))
    g = np.zeros((Nt + 1, Nobs))
    for n in range(1, Nt + 1):
        rhs = (s / dt) * p[n - 1] + q[n]
        p[n] = A_lu.solve(rhs)
        u[n] = K_lu.solve(G_op @ p[n])
        g[n] = E @ u[n]
    return dict(k=k, D=D, A_lu=A_lu, p=p, u=u, g=g)
 
 
def objective(m, d):
    st = forward(m)
    J = 0.0
    for n in range(1, Nt + 1):
        r = st["g"][n] - d[n]
        J += 0.5 * float(r @ r)
    return J, st
 
 
# ----------------------------------------------------------------------
# Tangent-linear operator L:  dm -> dg  (perturbation of g w.r.t. log-k)
#   d k    = k * dm
#   A dp^n = (M/dt) dp^{n-1} + (dD) p^n ,   dD = Dmat(dk)
#   du^n   = K^{-1} G_op dp^n ;  dg^n = E du^n
# ----------------------------------------------------------------------
def tangent(m, dm, st):
    k = st["k"]
    A_lu = st["A_lu"]
    dk = k * dm
    dD = Dmat(dk)
    dp = np.zeros((Nt + 1, Nn))
    dg = np.zeros((Nt + 1, Nobs))
    for n in range(1, Nt + 1):
        rhs = (s / dt) * dp[n - 1] + dD @ st["p"][n]
        dp[n] = A_lu.solve(rhs)
        dg[n] = E @ K_lu.solve(G_op @ dp[n])
    return dg
 
 
# ----------------------------------------------------------------------
# Adjoint operator L*:  data-space field w(t) -> model-space gradient
# Implements exactly the report's adjoint chain:
#   solid:  K^T u_dag^n = E^T w^n                     (strain-misfit dipole source)
#   fluid:  A^T lam^n  = (M/dt) lam^{n+1} + G_op^T u_dag^n   (backward in time)
#   grad :  dJ/dk_c = sum_n lam^n . (dD/dk_c) p^n     (correlation of grad p, grad p_dag)
#   chain:  dJ/dm_c = k_c * dJ/dk_c                   (since m = log k)
# ----------------------------------------------------------------------
def adjoint(m, w, st):
    k = st["k"]
    A_lu = st["A_lu"]
    p = st["p"]
 
    u_dag = np.zeros((Nt + 1, Ndof))
    for n in range(1, Nt + 1):
        u_dag[n] = KT_lu.solve(E.T @ w[n])
 
    lam = np.zeros((Nt + 2, Nn))                      # lam[Nt+1] = 0 (terminal cond.)
    A = ((s / dt) * sp.identity(Nn, format="csr") - st["D"])
    AT_lu = spla.splu(sp.csc_matrix(A.T))             # A^T (= A here, since A is symmetric)
    for n in range(Nt, 0, -1):
        rhs = (s / dt) * lam[n + 1] + (G_op.T @ u_dag[n])
        lam[n] = AT_lu.solve(rhs)
 
    grad_k = np.zeros(Nn)
    for c in range(Nn):
        acc = 0.0
        for n in range(1, Nt + 1):
            acc += lam[n] @ (Dc[c] @ p[n])
        grad_k[c] = acc
    grad_m = k * grad_k                               # chain rule for m = log k
    return grad_m
 
 
def gradient(m, d):
    J, st = objective(m, d)
    w = np.zeros((Nt + 1, Nobs))
    for n in range(1, Nt + 1):
        w[n] = st["g"][n] - d[n]                       # residual as the adjoint source
    g = adjoint(m, w, st)
    return J, g, st
 
 
# ----------------------------------------------------------------------
# Build a heterogeneous "true" model and a synthetic observation d.
# ----------------------------------------------------------------------
m_true = np.zeros(Nn)
for i in range(Nx):
    for j in range(Ny):
        # high-permeability SRV stripe near the middle column, tight elsewhere
        m_true[idx(i, j)] = (2.0 if abs(i - Nx // 2) <= 1 else -1.0) + 0.1 * (j - Ny / 2)
st_true = forward(m_true)
d = st_true["g"]
 
# Evaluation point (a different model), at which we verify the adjoint.
m0 = -0.5 + 0.3 * rng.standard_normal(Nn)
J0, st0 = objective(m0, d)
 
# ======================================================================
# TEST 1  -  Adjoint (dot-product) identity:  <L dm, w> == <dm, L* w>
# ======================================================================
print("=" * 64)
print("TEST 1  Adjoint identity (dot-product test):  <L dm, w> = <dm, L* w>")
print("=" * 64)
max_rel = 0.0
for trial in range(5):
    dm = rng.standard_normal(Nn)
    w  = rng.standard_normal((Nt + 1, Nobs)); w[0] = 0.0
    dg = tangent(m0, dm, st0)                          # L dm
    lhs = sum(float(dg[n] @ w[n]) for n in range(1, Nt + 1))   # <L dm, w>
    Lstar_w = adjoint(m0, w, st0)                      # L* w
    rhs = float(dm @ Lstar_w)                          # <dm, L* w>
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    max_rel = max(max_rel, rel)
    print(f"  trial {trial}:  <L dm,w> = {lhs: .8e}   <dm,L*w> = {rhs: .8e}   "
          f"rel.err = {rel: .2e}")
print(f"\n  worst relative error over trials: {max_rel:.2e}")
print("  -> PASS (adjoint identity holds to machine precision)\n"
      if max_rel < 1e-9 else "  -> FAIL\n")
 
# ======================================================================
# TEST 2  -  Taylor / central finite-difference gradient test
#   J(m+e dm) - J(m-e dm) = 2 e <g, dm> + O(e^3)
# ======================================================================
print("=" * 64)
print("TEST 2  Taylor gradient test:  central-FD slope vs adjoint <g, dm>")
print("=" * 64)
J0, g0, _ = gradient(m0, d)
dm = rng.standard_normal(Nn)
gdotdm = float(g0 @ dm)
print(f"  adjoint directional derivative <g, dm> = {gdotdm: .10e}\n")
print(f"  {'epsilon':>10} {'central-FD':>18} {'relative error':>16}")
prev = None
for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    Jp, _ = objective(m0 + e * dm, d)
    Jm, _ = objective(m0 - e * dm, d)
    fd = (Jp - Jm) / (2 * e)
    rel = abs(fd - gdotdm) / abs(gdotdm)
    print(f"  {e:>10.0e} {fd:>18.10e} {rel:>16.2e}")
print("\n  (relative error should shrink ~e^2 until rounding dominates -> gradient verified)")
 