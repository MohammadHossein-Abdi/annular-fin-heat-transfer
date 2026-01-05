import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# -------------------------------------------------
# Physical constants
# -------------------------------------------------
sigma = 5.670374419e-8    # Stefan–Boltzmann constant [W/m^2-K^4]
epsilon = 0.9            # emissivity (assumed, vacuum radiation)

# -------------------------------------------------
# Problem parameters
# -------------------------------------------------
k0 = 1.6372
alpha = 1.38505e-4

R0 = 0.05     # m
R  = 0.20     # m
delta = 0.005 # m

Tb_inner = 450.0  # K
Tb_outer = 350.0  # K
U = 300.0         # K

# -------------------------------------------------
# Discretization
# -------------------------------------------------
N = 120
r = np.linspace(R0, R, N)
dr = r[1] - r[0]

# -------------------------------------------------
# Thermal conductivity
# -------------------------------------------------
def k(T):
    return k0 * (1 + alpha * T)

# -------------------------------------------------
# Residual function
# -------------------------------------------------
def residual(T):
    res = np.zeros_like(T)

    for i in range(1, N - 1):
        k_plus = k(T[i + 1])
        k_minus = k(T[i - 1])

        conduction = (
            r[i + 1] * k_plus * (T[i + 1] - T[i])
            - r[i - 1] * k_minus * (T[i] - T[i - 1])
        ) / (r[i] * dr**2)

        radiation = (2 / delta) * epsilon * sigma * (T[i]**4 - U**4)

        res[i] = conduction - radiation

    # Boundary conditions
    res[0] = T[0] - Tb_inner
    res[-1] = T[-1] - Tb_outer

    return res

# -------------------------------------------------
# Initial guess
# -------------------------------------------------
T_guess = np.linspace(Tb_inner, Tb_outer, N)

# -------------------------------------------------
# Solve nonlinear system
# -------------------------------------------------
T_solution = fsolve(residual, T_guess)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure()
plt.plot(r, T_solution, linewidth=2)
plt.xlabel("Radius r (m)")
plt.ylabel("Temperature T (K)")
plt.title("Steady-State Temperature Distribution in Annular Fin")
plt.grid(True)
plt.show()
