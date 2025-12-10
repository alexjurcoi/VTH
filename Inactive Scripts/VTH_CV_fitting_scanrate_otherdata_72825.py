import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# Define your scan ID and its corresponding row and column info
scan_id = "05"
scan_locations = {
    "05": {"rows": (2402, 2752), "v_col": 7, "i_col": 9},
    # Add other scans here if needed
}

loc = scan_locations[scan_id]
start, stop = loc["rows"]
v_col, i_col = loc["v_col"], loc["i_col"]

# Load your experimental data file
filepath = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\UCSD_Data\10_CV_ScanRates_H2Sat_05_CV_C01.xlsx"
df = pd.read_excel(filepath, index_col=False)

# Extract voltage and current columns
V_exp = df.iloc[start:stop, v_col]
I_import = df.iloc[start:stop, i_col]

# Constants and Parameters
RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-9     # mol/cm²
conversion_factor = 1.60218e-19  # eV to J
AvoNum = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = [0.35, 0.5]
GHad_eV = -0.3
GHad = GHad_eV * AvoNum * conversion_factor  # Convert GHad from eV to J

# Mechanism: 0 = Volmer-Tafel, 1 = Volmer-Heyrovsky
mechanism_choice = 0
k_V_RDS = cmax * 10**3.7
if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
else:
    k_V = k_V_RDS
    k_H = k_V * 1000

# Voltage bounds
UpperV = 0
LowerV = -0.25

# Initial coverages
thetaA_H0 = 0.99
thetaA_Star0 = 1.0 - thetaA_H0
theta0 = np.array([thetaA_Star0, thetaA_H0])

# Functions
def eqpot(theta):
    thetaA_star, thetaA_H = theta
    U_V = (-GHad/F) + (RT * np.log(thetaA_star / thetaA_H)) / F
    U_H = 0
    if mechanism_choice == 1:
        U_H = GHad/F + (RT/F) * np.log(thetaA_H / thetaA_star)
    return U_V, U_H

def rates_r0(t, theta, potential):
    thetaA_star, thetaA_H = theta
    V = potential(t)
    U_V, U_H = eqpot(theta)
    r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * \
          (np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))
    r_T = 0
    if mechanism_choice == 0:
        r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * thetaA_star ** 2) * np.exp(-2 * GHad / RT))
    r_H = 0
    if mechanism_choice == 1:
        j1 = k_H * np.exp(-beta[1] * GHad / RT) * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
        r_H = j1 * (np.exp(-beta[1] * F * (V - U_H) / RT) - np.exp((1 - beta[1]) * F * (V - U_H) / RT))
    return r_V, r_T, r_H

def sitebal(t, theta, potential):
    r_V, r_T, r_H = rates_r0(t, theta, potential)
    if mechanism_choice == 0:
        return [(-r_V + 2 * r_T) / cmax, (r_V - 2 * r_T) / cmax]
    else:
        return [(r_H - r_V) / cmax, (r_V - r_H) / cmax]

# Scanrate simulation loop
scanrates = [0.05]
colors = ['r', 'g', 'purple']
model_data = []

# Example voltage mask range
mask_min = -0.25
mask_max = 0.00

scan_mask = (V_exp <= mask_max) & (V_exp >= mask_min)
V_exp_masked = V_exp[scan_mask].reset_index(drop=True)
I_import_masked = I_import[scan_mask].reset_index(drop=True)

adjustment_exp = 0.0537  # your known offset
I_exp_masked = (I_import_masked / 0.0929) + adjustment_exp

# --- R² function ---
def r_squared(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    ss_res = np.sum((data1 - data2) ** 2)
    ss_tot = np.sum((data1 - np.mean(data1)) ** 2)
    return 1 - (ss_res / ss_tot)

# --- Model simulation loop ---
fixed_scanrate = 0.050  # or whatever value you want
GHad_eVs = [-0.3]  # example values to sweep over
k = [cmax * 10 ** 3.5, cmax * 10 ** 3.7, cmax * 10 ** 3.9]

colors = ['r', 'g', 'purple', 'orange', 'teal']
model_data = []

def make_potential(scanrate):
    def potential(x):
        single_sweep_time = (UpperV - LowerV) / scanrate
        cycle_time = 2 * single_sweep_time
        t_in_cycle = x % cycle_time
        if t_in_cycle < single_sweep_time:
            return UpperV - scanrate * t_in_cycle
        else:
            return LowerV + scanrate * (t_in_cycle - single_sweep_time)
    return potential

for k_val, color in zip(k, colors):
    k_V = k_val
    k_T = k_V * 1000
    potential = make_potential(fixed_scanrate)
    timescan = 2 * (UpperV - LowerV) / fixed_scanrate
    t = np.arange(0.0, timescan, fixed_scanrate)
    duration = [0, t[-1]]

    sol = solve_ivp(lambda t, theta: sitebal(t, theta, potential), duration, theta0, t_eval=t, method='BDF')
    r0_vals = np.array([rates_r0(ti, th, potential) for ti, th in zip(t, sol.y.T)])
    volmer_rate = r0_vals[:, 0]
    curr_model = volmer_rate * -F * 1000
    V_model = np.array([potential(ti) for ti in t])

    # Mask and interpolate
    model_mask = (V_model <= mask_max) & (V_model >= mask_min)
    V_model_masked = V_model[model_mask]
    curr_model_masked = curr_model[model_mask]

    interp_func = interp1d(V_model_masked, curr_model_masked, kind='linear', fill_value='extrapolate')
    I_model_interp = interp_func(V_exp_masked)

    cut = 5
    r2_val = r_squared(I_exp_masked[cut:], I_model_interp[cut:])
    model_data.append((V_model[cut:], curr_model[cut:], k_val, r2_val, color))
    # Collect R² vs GHad data
    k_list = [k for _, _, k, _, _ in model_data]
    r2_list = [r2 for _, _, _, r2, _ in model_data]

fig, ax = plt.subplots(figsize=(8, 10))

for V_model, curr_model, k_val, r2, color in model_data:
    k_display = k_val  # or k_val / cmax if your k_list was in normalized form
    ax.plot(V_model, curr_model, color=color,
            label=f'k = {k_display:.2e} (R² = {r2:.4f})')

ax.plot(V_exp_masked, I_exp_masked, 'b', label='Experimental Data')
ax.set_xlabel('Voltage vs. RHE (V)')
ax.set_ylabel('Kinetic current (mA/cm²)')
ax.set_title(f'Kinetic Current vs Voltage at Various K')
ax.grid()
ax.legend()
plt.show()

fig, ax_r2 = plt.subplots(figsize=(6, 4))
ax_r2.plot(k_list, r2_list, marker='o', linestyle='-', color='black')
ax_r2.set_xlabel(r'K / cmax')
ax_r2.set_ylabel(r'$R^2$')
ax_r2.set_title(f'$R^2$ vs K')
#ax_r2.set_ylim(0, 1)
ax_r2.ticklabel_format(style='plain', axis='y')
ax_r2.grid(True)
plt.tight_layout()
plt.show()
