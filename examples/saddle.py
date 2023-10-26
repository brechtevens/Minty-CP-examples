#%% Preamble

import sympy as sym
from IPython.display import display, Math

#%% Set up problem variables

a, b, c, l = sym.symbols('a,b,c,l', real=True)
γ, λ = sym.symbols('γ,λ', real=True, positive=True)
τ = 1/(γ*l**2)

A = sym.Matrix([[0, a], [-a, 0]])
B = sym.Matrix([[b, 0, 0], [0, b, 0], [0, 0, c]])
L = sym.Matrix([[l, 0],[0, l], [0, 0]])
P = sym.Matrix([[γ**-1*sym.Matrix(sym.eye(2)), -L.T], [-L, τ**-1*sym.Matrix(sym.eye(3))]])

Tp = A + L.T @ B @ L
Td = L @ A.inv() @ L.T + B.inv()
Tpd = sym.Matrix([[A, L.T], [-L, B**-1]])

U = sym.Matrix([
    [1/sym.sqrt(1+γ**2*l**2)*sym.eye(2), sym.zeros(2,1)],
    [-γ*l/sym.sqrt(1+γ**2*l**2)*sym.eye(2), sym.zeros(2,1)],
    [sym.zeros(1,2), sym.eye(1)]
    ])

beta_P, beta_D, beta_P_prime, beta_D_prime = sym.symbols('beta_P, beta_D, beta_P^\', beta_D^\'', real=True)
m, n = L.shape
V = sym.Matrix([[beta_P*L.pinv() @ L + beta_P_prime * (sym.eye(n) - L.pinv() @ L), sym.zeros(n,m)], [sym.zeros(m,n), beta_D * L @ L.pinv() + beta_D_prime * (sym.eye(m) - L @ L.pinv())]])
    
#%% Example 3.7(i)
#%% Part 1: find interval for λ such that the absolute value of the eigenvalues of H are smaller than 1
H = sym.simplify(sym.eye(5) + λ*((P + Tpd).inv() @ P - sym.eye(5)))

for ev in H.eigenvals():
    aev = sym.re(ev)**2 + sym.im(ev)**2
    display(Math('\lambda \in (0, %s)' %sym.latex(sym.solve(sym.Eq(aev, 1), λ)[0])))

#%% Part 2: find interval for λ such that the absolute value of the eigenvalues of proj_{range(P)} H are smaller than 1
H_proj = sym.simplify(U @ U.T @ H)

for ev in H_proj.eigenvals():
    aev = sym.re(ev)**2 + sym.im(ev)**2
    if aev != 0:
        display(Math('\lambda \in (0, %s)' %sym.latex(sym.solve(sym.Eq(aev, 1), λ)[0])))

# %% Example 3.7(ii)
LMI = sym.simplify(U.T @ Tpd.inv().T @ ((Tpd + Tpd.T)/2 - Tpd.T * V * Tpd) @ Tpd.inv() @ U)
display(LMI)

# %% Example 3.7(iii)

display(
    Math('trace(T_P) = %s' %sym.latex(sym.trace(Tp.subs(a,10).subs(b,-1/4).subs(c,-1/4).subs(l,2)))),
    Math('trace(T_D) = %s' %sym.latex(sym.trace(Td.subs(a,10).subs(b,-1/4).subs(c,-1/4).subs(l,2)))),
    Math('trace(T_{PD}) = %s' %sym.latex(sym.trace(Tpd.subs(a,10).subs(b,-1/4).subs(c,-1/4).subs(l,2))))
)

# %%
