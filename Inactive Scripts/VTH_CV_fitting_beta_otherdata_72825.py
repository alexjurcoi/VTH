import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# === Load experimental data ===
scan_id = "05"
scan_locations = {
    "05": {"rows": (2402, 2752), "v_col": 7, "i_col": 9},
}
loc = scan_locations[scan_id]
start, stop = loc["rows"]
v_col, i_col = loc["v_col"], loc["i_col"]

filepath = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\UCSD_Data\10_CV_ScanRates_H2Sat_05_CV_C01.xlsx"
df = pd.read_excel(filepath, index_col=False)

V_exp = df.iloc[start:stop, v_col]
I_import = df.iloc[start:stop, i_col]

# === Constants ===
RT = 8.314 * 298
F = 96485.0
cmax = 7.5e-9
conversion_factor = 1.60218e-19
AvoNum = 6.02e23
partialPH2 = 1.0
GHad_eV = -0.3
GHad = GHad_eV * AvoNum * conversion_factor
mechanism_choice = 0  # Volmer-Tafel
k_V = cmax * 10**3.7
k_T = k_V * 1000
fixed_scanrate = 0.05
UpperV = 0
LowerV = -0.25

# === Initial Conditions ===
thetaA_H0 = 0.99
thetaA_Star0 = 1.0 - thetaA_H0
theta0 = np.array([thetaA_Star0, thetaA_H0])

# === Potential function ===
def make_potential(scanrate):
    def potential(t):
        single_sweep_time = (UpperV - LowerV) / scanrate
        cycle_time = 2 * single_sweep_time
        t_in_cycle = t % cycle_time
        if t_in_cycle < single_sweep_time:
            return UpperV - scanrate * t_in_cycle
        else:
            return LowerV + scanrate * (t_in_cycle - single_sweep_time)
    return potential

# === Equilibrium potential ===
def eqpot(theta):
    theta_star, theta_H = theta
    U_V = (-GHad / F) + (RT / F) * np.log(theta_star / theta_H)
    U_H = 0
    return U_V, U_H

# === Rate expressions ===
def rates_r0(t, theta, potential, beta):
    theta_star, theta_H = theta
    V = potential(t)
    U_V, U_H = eqpot(theta)
    r_V = k_V * (theta_star ** (1 - beta)) * (theta_H ** beta) * np.exp(beta * GHad / RT) * \
          (np.exp(-beta * F * (V - U_V) / RT) - np.exp((1 - beta) * F * (V - U_V) / RT))
    r_T = k_T * ((theta_H ** 2) - (partialPH2 * theta_star ** 2) * np.exp(-2 * GHad / RT))
    return r_V, r_T

def sitebal(t, theta, potential, beta):
    r_V, r_T = rates_r0(t, theta, potential, beta)
    return [(-r_V + 2 * r_T) / cmax, (r_V - 2 * r_T) / cmax]

# === R² calculation ===
def r_squared(data1, data2):
    ss_res = np.sum((data1 - data2) ** 2)
    ss_tot = np.sum((data1 - np.mean(data1)) ** 2)
    return 1 - (ss_res / ss_tot)

# === Mask experimental data ===
mask_min = -0.25
mask_max = 0.00
scan_mask = (V_exp <= mask_max) & (V_exp >= mask_min)
V_exp_masked = V_exp[scan_mask].reset_index(drop=True)
I_import_masked = I_import[scan_mask].reset_index(drop=True)
adjustment_exp = 0.0537
I_exp_masked = (I_import_masked / 0.0929) + adjustment_exp

# === Main loop: vary beta ===
beta_vals = [0.3, 0.35, 0.4]
colors = ['r', 'g', 'purple']
model_data = []

potential = make_potential(fixed_scanrate)
timescan = 2 * (UpperV - LowerV) / fixed_scanrate
t = np.arange(0.0, timescan, fixed_scanrate)
duration = [0, t[-1]]

for beta_val, color in zip(beta_vals, colors):
    sol = solve_ivp(lambda t, theta: sitebal(t, theta, potential, beta_val),
                    duration, theta0, t_eval=t, method='BDF')
    r0_vals = np.array([rates_r0(ti, th, potential, beta_val)[0] for ti, th in zip(t, sol.y.T)])
    curr_model = r0_vals * -F * 1000  # mA/cm²
    V_model = np.array([potential(ti) for ti in t])

    model_mask = (V_model <= mask_max) & (V_model >= mask_min)
    V_model_masked = V_model[model_mask]
    curr_model_masked = curr_model[model_mask]

    interp_func = interp1d(V_model_masked, curr_model_masked, kind='linear', fill_value='extrapolate')
    I_model_interp = interp_func(V_exp_masked)

    cut = 5
    r2_val = r_squared(I_exp_masked[cut:], I_model_interp[cut:])
    model_data.append((V_model[cut:], curr_model[cut:], beta_val, r2_val, color))

# === Plot current vs voltage ===
fig, ax = plt.subplots(figsize=(8, 6))
for V_model, curr_model, beta_val, r2_val, color in model_data:
    ax.plot(V_model, curr_model, color=color,
            label=f'β = {beta_val:.2f} (R² = {r2_val:.4f})')
ax.plot(V_exp_masked, I_exp_masked, 'b', label='Experimental Data')
ax.set_xlabel('Voltage vs. RHE (V)')
ax.set_ylabel('Kinetic current (mA/cm²)')
ax.set_title('Kinetic Current vs Voltage at Various β')
ax.grid()
ax.legend()
plt.tight_layout()
plt.show()

# === Plot R² vs beta ===
fig, ax_r2 = plt.subplots(figsize=(6, 4))
beta_list = [beta for _, _, beta, _, _ in model_data]
r2_list = [r2 for _, _, _, r2, _ in model_data]
ax_r2.plot(beta_list, r2_list, marker='o', linestyle='-', color='black')
ax_r2.set_xlabel(r'Volmer $\beta$')
ax_r2.set_ylabel(r'$R^2$')
ax_r2.set_title(r'$R^2$ vs Volmer $\beta$')
ax_r2.grid(True)
plt.tight_layout()
plt.show()
