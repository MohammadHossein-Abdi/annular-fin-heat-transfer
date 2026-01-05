import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==================================================
# COMMON PARAMETERS
# ==================================================
rho = 2700.0
cp  = 900.0
k0  = 1.6372
alpha = 1.38505e-4

sigma = 5.670374419e-8
epsilon = 0.9

R0 = 0.05
R  = 0.20
delta = 0.005
U = 300.0

N = 100
r = np.linspace(R0, R, N)
dr = r[1] - r[0]

# ==================================================
# BASE TEMPERATURE
# ==================================================
def T_base(t):
    t_min = t / 60.0
    if t_min <= 15.0:
        return 540.0 - 6.0 * t_min
    else:
        return 450.0

# ==================================================
# FULL MODEL (Part B)
# ==================================================
def k(T):
    return k0 * (1 + alpha * T)

def dTdt_full(t, T):
    dT = np.zeros_like(T)

    for i in range(1, N - 1):
        kp = k(T[i + 1])
        km = k(T[i - 1])

        conduction = (
            r[i + 1] * kp * (T[i + 1] - T[i])
            - r[i - 1] * km * (T[i] - T[i - 1])
        ) / (r[i] * dr**2)

        radiation = (2 / delta) * epsilon * sigma * (T[i]**4 - U**4)

        dT[i] = (conduction - radiation) / (rho * cp)

    # Inner boundary
    T[0] = T_base(t)
    dT[0] = 0.0

    # Outer radiative boundary
    dTdr_R = -(sigma / k(T[-1])) * (T[-1]**4 - U**4)
    T[-1] = T[-2] + dr * dTdr_R
    dT[-1] = 0.0

    return dT

# ==================================================
# VALIDATION MODEL (constant k, linear radiation)
# ==================================================
T_mean = 400.0
h_r = 4 * sigma * T_mean**3

def dTdt_validation(t, T):
    dT = np.zeros_like(T)

    for i in range(1, N - 1):
        conduction = (
            r[i + 1] * (T[i + 1] - T[i])
            - r[i - 1] * (T[i] - T[i - 1])
        ) / (r[i] * dr**2)

        radiation = (2 * h_r / delta) * (T[i] - U)

        dT[i] = (k0 * conduction - radiation) / (rho * cp)

    # Inner boundary
    T[0] = T_base(t)
    dT[0] = 0.0

    # Outer boundary
    dTdr_R = -(h_r / k0) * (T[-1] - U)
    T[-1] = T[-2] + dr * dTdr_R
    dT[-1] = 0.0

    return dT

# ==================================================
# TIME INTEGRATION
# ==================================================
T0 = np.ones(N) * 540.0
t_span = (0, 3600)
t_eval = np.linspace(0, 3600, 200)

sol_full = solve_ivp(dTdt_full, t_span, T0, t_eval=t_eval, method="BDF")
sol_val  = solve_ivp(dTdt_validation, t_span, T0, t_eval=t_eval, method="BDF")

# ==================================================
# AUTOMATIC COMPARISON PLOTS
# ==================================================
compare_times = [5, 15, 30, 60]  # minutes

plt.figure(figsize=(8, 6))

for t_min in compare_times:
    idx = np.argmin(np.abs(sol_full.t - t_min * 60))

    plt.plot(
        r,
        sol_full.y[:, idx],
        linestyle="-",
        label=f"Full model, t={t_min} min"
    )

    plt.plot(
        r,
        sol_val.y[:, idx],
        linestyle="--",
        label=f"Validation, t={t_min} min"
    )

plt.xlabel("Radius r (m)")
plt.ylabel("Temperature T (K)")
plt.title("Comparison: Full Model vs Validation Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
