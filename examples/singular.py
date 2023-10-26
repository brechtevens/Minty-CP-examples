#%% Preamble

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

#%% Set up problem variables

l2 = sym.symbols('l_2', real=True)
l3 = 0.2
a2 = 1 + sym.sqrt(1-l2**2)
a3 = 1 + sym.sqrt(1-l3**2)
b2 = 1/a2
b3 = 1/a3

γ = 1
τ = 1
λ = sym.symbols('λ', real=True, positive=True)

A = sym.Matrix([[1, 0, 0],[0, a2, 0], [0, 0, a3]])
B = sym.Matrix([[1, 0, 0],[0, b2, 0], [0, 0, b3]])
L = sym.Matrix([[1, 0, 0],[0, l2, 0], [0, 0, l3]])

m, n = L.shape

P = sym.Matrix([[γ**-1*sym.eye(3), -L.T], [-L, τ**-1*sym.eye(3)]])
Tpd = sym.Matrix([[A, L.T], [-L, B**-1]])

U = sym.Matrix([
    [1/sym.sqrt(1+γ**2)*sym.eye(1), sym.zeros(1,4)],
    [sym.zeros(2,1), sym.diag(l2/abs(l2), l3/abs(l3)), sym.zeros(2,2)],
    [-γ/sym.sqrt(1+γ**2)*sym.eye(1), sym.zeros(1,4)],
    [sym.zeros(2,3), sym.eye(2)]
    ])

#%% Figure 1

l2_num = 0.5
A_num = np.array(A.subs(l2, l2_num)).astype(np.float64)
Binv_num = np.linalg.inv(np.array(B.subs(l2, l2_num)).astype(np.float64))
L_num = np.array(L.subs(l2, l2_num)).astype(np.float64)

def resolvent_A(γ, x):
    return np.linalg.solve(np.eye(n) + γ * A_num, x)

def resolvent_Binv(γ, x):
    return np.linalg.solve(np.eye(m) + γ * Binv_num, x)

def cp(x0, y0, λ, N):
    zx = np.NaN * np.ones((N, n))
    zy = np.NaN * np.ones((N, m))
    z̅x = np.NaN * np.ones((N, n))
    z̅y = np.NaN * np.ones((N, m))

    x = x0
    y = y0
    for i in range(N):
        zx[i] = x.T
        zy[i] = y.T
        x̅ = resolvent_A(γ, x - γ*L_num.T @ y)
        z̅x[i] = x̅.T
        y̅ = resolvent_Binv(τ, y + τ*L_num @ (2*x̅ - x))
        z̅y[i] = y̅.T
        x = x + λ * (x̅ - x)
        y = y + λ * (y̅ - y)

    return zx, zy, z̅x, z̅y

N = 30
λ_num = 2.1

x0 = np.array([[-1], [-1], [-1/2]])
y0 = np.array([[1], [1/2], [3]])
zx, zy, z̅x, z̅y = cp(x0, y0, λ_num, N)
zz = np.hstack([zx, zy])

plt.figure()
plt.semilogy(np.linalg.norm(zz, ord=2, axis=1))
plt.title(r'$||z^k||$')

plt.figure()
plt.semilogy(np.abs(zx[:14,0] - zy[:14,0]))
plt.title(r'$||X_1^T x^k - Y_1^T y^k||$')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(zx[:6, 0], zx[:6, 1], zx[:6, 2], 'b', label=r'$x^k$')
ax.scatter3D(zx[:6, 0], zx[:6, 1], zx[:6, 2], c='b')
ax.plot3D(0*zx[:6, 0], zx[:6, 1], zx[:6, 2], 'k', label=r'$X_{2:}^T x^k$')
ax.scatter3D(0*zx[:6, 0], zx[:6, 1], zx[:6, 2], c='k')
plt.legend()
plt.title('primal sequences')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(zy[:6, 0], zy[:6, 1], zy[:6, 2], 'b', label=r'$y^k$')
ax.scatter3D(zy[:6, 0], zy[:6, 1], zy[:6, 2], c='b')
ax.plot3D(0*zy[:6, 0], zy[:6, 1], zy[:6, 2], 'k', label=r'$Y_{2:}^T y^k$')
ax.scatter3D(0*zy[:6, 0], zy[:6, 1], zy[:6, 2], c='k')
plt.legend()
plt.title('dual sequences')
plt.show()

#%% Figure 2

H = sym.simplify(U @ U.T @ sym.simplify(sym.eye(6) + λ*((P + Tpd).inv() @ P - sym.eye(6))))
squared = sym.simplify(sym.expand(sym.Matrix([ev**2 for ev in list(H.eigenvals().keys())])))
upper_bounds = sym.Matrix([sym.solve(sym.Eq(eig, 1), λ) for eig in squared])

l2_list = np.arange(0.0,1,0.01)
lambda_max = []

for l2_num in l2_list:
    lambda_max.append(min(upper_bounds.subs(l2, l2_num)))

beta_P = 1/2
beta_D = 1/2
etabar_thm = lambda l2 : 1 + (1/(2*γ))*beta_P + (1/(2*τ))*beta_D - np.sqrt(((1/(2*γ))*beta_P - (1/(2*τ))*beta_D)**2 + beta_P*beta_D*max(abs(l2), abs(l3))**2)

lambda_max_thm = []

for l2_num in l2_list:
    lambda_max_thm.append(2*etabar_thm(l2_num))

plt.figure()
plt.plot(l2_list, lambda_max, label=r'$\bar{\lambda}_{\rm spectral}$')
plt.plot(l2_list, lambda_max_thm, label=r'$\bar{\lambda}$')
plt.xlabel(r'$\ell_2$')
plt.legend()
plt.show()

# %%
