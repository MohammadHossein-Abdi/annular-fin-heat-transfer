import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --------------------------------------------------
# Physical constants
# --------------------------------------------------
sigma = 5.670374419e-8     # Stefan–Boltzmann constant [W/m^2-K^4]
epsilon = 0.9             # emissivity (vacuum radiation)

rho = 2700.0              # density [kg/m^3]
cp  = 900.0               # specific heat [J/kg-K]

k0 = 1.6372
alpha = 1.38505e-4

# --------------------------------------------------
# Geometry and conditions
# --------------------------------------------------
R0 = 0.05      # inner radius [m]
R  = 0.20      # outer radius [m]
delta = 0.005  # thickness [m]

U = 300.0      # surrounding temperature [K]

# --------------------------------------------------
# Spatial grid
# --------------------------------------------------
N = 120
r = np.linspace(R0, R, N)
dr = r[1] - r[0]

# --------------------------------------------------
# Thermal conductivity
# --------------------------------------------------
def k(T):
    return k0 * (1 + alpha * T)

# --------------------------------------------------
# Base temperature (startup condition)
# --------------------------------------------------
def T_base(t):
    t_min = t / 60.0   # convert seconds to minutes
    if t_min <= 15.0:
        return 540.0 - 6.0 * t_min
    else:
        return 450.0

# --------------------------------------------------
# PDE → ODE system
# --------------------------------------------------
def dTdt(t, T):
    dT = np.zeros_like(T)

    for i in range(1, N - 1):
        k_plus = k(T[i + 1])
        k_minus = k(T[i - 1])

        conduction = (
            r[i + 1] * k_plus * (T[i + 1] - T[i])
            - r[i - 1] * k_minus * (T[i] - T[i - 1])
        ) / (r[i] * dr**2)

        radiation = (2 / delta) * epsilon * sigma * (T[i]**4 - U**4)

        dT[i] = (conduction - radiation) / (rho * cp)

    # ----------------------------
    # Inner boundary (Dirichlet)
    # ----------------------------
    T[0] = T_base(t)
    dT[0] = 0.0

    # ----------------------------
    # Outer boundary (radiative)
    # -k dT/dr = σ(T⁴ - U⁴)
    # ----------------------------
    dTdr_R = -(sigma / k(T[-1])) * (T[-1]**4 - U**4)
    T[-1] = T[-2] + dr * dTdr_R
    dT[-1] = 0.0

    return dT

# --------------------------------------------------
# Initial condition
# --------------------------------------------------
T0 = np.ones(N) * 540.0

# --------------------------------------------------
# Time integration
# --------------------------------------------------
t_span = (0, 3600)  # 1 hour [s]
t_eval = np.linspace(0, 3600, 200)

solution = solve_ivp(dTdt, t_span, T0, t_eval=t_eval, method='BDF')

# --------------------------------------------------
# Plot temperature distribution at selected times
# --------------------------------------------------
plt.figure()
for idx in [0, 50, 100, 199]:
    plt.plot(r, solution.y[:, idx], label=f"t = {solution.t[idx]/60:.1f} min")

plt.xlabel("Radius r (m)")
plt.ylabel("Temperature T (K)")
plt.title("Transient Temperature Distribution Along Fin Radius")
plt.legend()
plt.grid(True)
plt.show()
