from scipy.optimize import fsolve
import numpy as np

# Constants
RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
Avo = 6.02e23
conversion_factor = 1.60218e-19  # eV to J

# ΔG_H in eV
dG_eV = 0.10
dG_J = dG_eV * Avo * conversion_factor

# Langmuir isotherm
def theta_H(U, dG):
    return 1 / (1 + np.exp((F * U + dG) / RT))

# Self-consistent equation: f(U) = 0
def f(U, dG):
    theta = theta_H(U, dG)
    lhs = U
    rhs = -dG / F + (RT / F) * np.log(theta / (1 - theta))
    return lhs - rhs

# Initial guess for U_eq
U_guess = -0.1

# Solve
U_eq = fsolve(f, U_guess, args=(dG_J,))[0]

print(f"Equilibrium potential for GHad = {dG_eV:.2f} eV is: {U_eq:.4f} V vs. RHE")

# Optional: Also print the equilibrium theta_H
theta_eq = theta_H(U_eq, dG_J)
print(f"Equilibrium theta_H: {theta_eq:.4f}")
