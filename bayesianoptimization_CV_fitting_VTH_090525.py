# ===================== BAYES OPT FOR VT (k_V, beta_V) per GHad =====================
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
HAVE_SKOPT = True

# Ask user for mechanism choice
while True:
    mechanism_choice = input(
        "Choose mechanism:\n"
        f"{'Volmer RDS Tafel Fast (0)'.rjust(40)}\n"
        f"{'Volmer RDS Heyrovsky Fast (1)'.rjust(40)}\n")
    if mechanism_choice in ["0", "1"]:
        break  # Exit the loop if input is valid
    print("Invalid choice. Please enter 0 or 1.")

# Convert to integer for logic checks
mechanism_choice = int(mechanism_choice)

RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-9     # mol/cm²
conversion_factor = 1.60218e-19  # eV to J
AvoNum = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = [0.5, 0.5]
GHad_eV = -0.3

k_V_RDS = cmax * 10**3.7

if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
elif mechanism_choice == 1:
    k_V = k_V_RDS
    k_H = k_V * 1000

GHad = GHad_eV * AvoNum * conversion_factor  # Convert GHad from eV to J

# # potential sweep & time
UpperV = 0
LowerV = -0.30
scanrate = 0.05  #scan rate in V/s
timescan = 2*(UpperV-LowerV)/(scanrate)
max_time = 20
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

# Keep your original k_V_RDS definition
# k_V will be optimized; Tafel rate uses k_T = 1000 * k_V (retain your relationship)

MECH_NAME = "VT" if mechanism_choice == 0 else "VH"

scan_locations = {
    "02": {"rows": (311, 746),  "v_col": 7, "i_col": 9},
    "03": {"rows": (2137, 2463), "v_col": 26, "i_col": 25},
    "04": {"rows": (2462, 2719), "v_col": 7, "i_col": 9},
    "05": {"rows": (2402, 2752), "v_col": 7, "i_col": 9},
    "06": {"rows": (2477, 2837), "v_col": 7, "i_col": 9},
    "07": {"rows": (2569, 2941), "v_col": 7, "i_col": 9},
    "08": {"rows": (2595, 2968), "v_col": 1, "i_col": 2},
}
scan_id = "05"


loc = scan_locations[scan_id]
start, stop = loc["rows"]
v_col, i_col = loc["v_col"], loc["i_col"]

filepath = fr"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\UCSD_Data\10_CV_ScanRates_H2Sat_05_CV_C01.xlsx"
df = pd.read_excel(filepath, index_col=False)

V_exp = df.iloc[start:stop, v_col]
I_import = df.iloc[start:stop, i_col]

print(df.head(15))      # first rows
print(df.tail(15))      # last rows
print(df.iloc[2137:2463, [1,2]].head())   # the range and columns you used for scan 03
print("V_exp min:", V_exp.min(), "max:", V_exp.max())

print("V_exp min:", V_exp.min(), "max:", V_exp.max())

# Fixed fitting window
FIT_MIN, FIT_MAX = -0.25, -0.01

# After you set V_exp and I_import (and convert to volts if needed)
scan_mask = (V_exp >= FIT_MIN) & (V_exp <= FIT_MAX)
V_exp_masked = V_exp[scan_mask].reset_index(drop=True)
I_import_masked = I_import[scan_mask].reset_index(drop=True)

# Hard guard: must have enough points in the window
if len(V_exp_masked) < 8:
    raise ValueError(f"No/too few experimental points in [{FIT_MIN}, {FIT_MAX}] V "
                     f"(found {len(V_exp_masked)}). Check columns/units/mask range.")

# Normalize by area and baseline
I_exp_area = I_import_masked / 0.0929
adjustment_exp = -I_exp_area.iloc[0]       # <-- iloc (not [0])
I_exp_masked = I_exp_area + adjustment_exp


# Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    #timescan above is the same as single_sweep_time
    single_sweep_time = (UpperV - LowerV) / scanrate
    cycle_time = 2 * single_sweep_time

    t_in_cycle = x % cycle_time

    if t_in_cycle < single_sweep_time: #forward
        return UpperV - scanrate * t_in_cycle
    else: #reverse
        return LowerV + scanrate * (t_in_cycle - single_sweep_time)

#Function to calculate U and Keq from theta, dG
def eqpot(theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta # unpack surface coverage

    ##Volmer
    U_V = 0
    U_V = (-GHad/F) + (RT*np.log(thetaA_star/thetaA_H))/F
    #U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)

    ##Heyrovsky
    U_H = 0
    if mechanism_choice == 1:
        U_11 = GHad/F
        U_12 = (RT/F) * np.log(thetaA_H/thetaA_star)
        U_H = U_11 + U_12

    return U_V, U_H

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U_V, U_H = eqpot(theta) #call function to find U for given theta

    ##Volmer Rate Equation
    r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * (np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

    ##Tafel Rate Equation
    r_T = 0
    if mechanism_choice == 0:
        r_T = k_T * ((thetaA_H **2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2*GHad) / RT)))

    ##Heyrovsky Rate Equation
    r_H = 0
    if mechanism_choice == 1:
        j1 = k_H  *  np.exp(-beta[1]*GHad/RT)  *  thetaA_star**beta[1]  *  thetaA_H**(1-beta[1])
        exp21 = np.exp(-beta[1] * F * (V-U_H) / RT)
        exp22 = np.exp((1-beta[1]) * F * (V-U_H) / RT)
        r_H = j1 * (exp21 - exp22)

    return r_V, r_T, r_H

def sitebal(t, theta):
    r_V, r_T, r_H = rates_r0(t, theta)
    if mechanism_choice in [0]:
        thetaStar_rate_VT = (-r_V + (2*r_T)) / cmax
        thetaH_rate_VT = (r_V - (2*r_T)) / cmax
        dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT] # [0 = star, 1 = H]
    elif mechanism_choice in [1]:
        theta_star_rate = r_H-r_V      # summing all step rates based on how they affect theta_star
        theta_H_rate = r_V-r_H        # summing all step rates based on how they affect theta_H
        dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
    return dthetadt

# Supply the list of GHad values (eV) to fit over:
GHad_eV_list = np.linspace(-0.37, -0.3, 20)
def simulate_and_score(kV_rel_log10, betas, GHad_eV_val, kT_rel_log10=None):
    global k_V, k_T, k_H, beta, GHad_eV, GHad

    PENALTY = 1e6
    try:
        # ---- parameters from optimizer ----
        k_V = k_V_RDS * (10.0 ** float(kV_rel_log10))
        beta = [float(betas[0]), float(betas[1])]

        if mechanism_choice == 0:   # VT
            if kT_rel_log10 is None:
                k_T = 1000.0 * k_V  # fallback
            else:
                k_T = 10.0 ** float(kT_rel_log10) 
        else:                       # VH
            k_H = 1000.0 * k_V

        GHad_eV = float(GHad_eV_val)
        GHad = GHad_eV * AvoNum * conversion_factor

        # ---- simulate ----
        sol = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
        if (not sol.success) or (sol.y.shape[1] != len(t)):
            return PENALTY

        rates = np.array([rates_r0(tt, th) for tt, th in zip(t, sol.y.T)])
        I_model_full = rates[:, 0] * -F * 1000.0
        V_model_full = np.array([potential(tt) for tt in t])

        # same fixed window as manual code
        model_mask = (V_model_full >= FIT_MIN) & (V_model_full <= FIT_MAX)
        if model_mask.sum() < 8:
            return PENALTY

        V_model_masked = V_model_full[model_mask]
        I_model_masked = I_model_full[model_mask]
        interp = interp1d(V_model_masked, I_model_masked, kind='linear',
                          fill_value='extrapolate', assume_sorted=False)
        I_model_on_exp = interp(V_exp_masked.to_numpy())

        eps = 1e-12
        y_true = np.log10(np.clip(np.abs(I_exp_masked.to_numpy()), eps, None))
        y_pred = np.log10(np.clip(np.abs(I_model_on_exp), eps, None))

        # sanity
        if (not np.all(np.isfinite(y_true))) or (not np.all(np.isfinite(y_pred))):
            return PENALTY

        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot <= 0 or not np.isfinite(ss_tot):
            return PENALTY

        r2 = 1.0 - (ss_res / ss_tot)
        if not np.isfinite(r2):
            return PENALTY

        return 1.0 - float(r2)   # <<< minimize loss

    except Exception:
        return PENALTY

results = []
print(f"\n=== {('Volmer–Tafel' if mechanism_choice==0 else 'Volmer–Heyrovsky')} parameter fitting per GHad ===")

for g in GHad_eV_list:
    print(f"\nOptimizing for GHad = {g:.3f} eV ...")

    # ----- define search space + objective -----
    if mechanism_choice == 0:   # VT: optimize kV, beta_V, kT (independent of kV)
        space = [
            Real(-5.0,  3.0,  name="kV_rel_log10"),
            Real( 0.2, 0.8, name="beta_V"),
            Real(-5.0,  3.0,  name="kT_log10"),
        ]
        def obj(x):  # x = [kV_rel_log10, beta_V, kT_log10]
            return simulate_and_score(
                x[0], [x[1], beta[1]], g, kT_rel_log10=x[2]
            )
    else:                       # VH: optimize kV, beta_V, beta_H
        space = [
            Real(-3.0, 3.0,  name="kV_rel_log10"),
            Real( 0.2, 0.9,  name="beta_V"),
            Real( 0.2, 0.9,  name="beta_H"),
        ]
        def obj(x):  # x = [kV_rel_log10, beta_V, beta_H]
            return simulate_and_score(
                x[0], [x[1], x[2]], g
            )

    # ----- run BO -----
    res = gp_minimize(
        obj, space,
        n_calls=20, n_initial_points=20,
        initial_point_generator="lhs",
        acq_func="EI", random_state=0
    )

    # ----- unpack + append (robust to VT/VH) -----
    try:
        best_rel   = res.x[0]
        best_kV    = k_V_RDS * (10.0 ** best_rel)
        best_betaV = res.x[1]
        best_R2    = 1.0 - res.fun

        if mechanism_choice == 0:
            best_kT = 10.0 ** res.x[2]
            print(f"  -> best k_V = {best_kV:.6g}, beta_V = {best_betaV:.3e}, k_T = {best_kT:.3e}, R² = {best_R2:.4f}")
            results.append((g, best_kV, best_betaV, best_kT, best_R2))
        else:
            best_betaH = res.x[2]
            print(f"  -> best k_V = {best_kV:.6g}, beta_V = {best_betaV:.3e}, beta_H = {best_betaH:.3f}, R² = {best_R2:.4f}")
            results.append((g, best_kV, best_betaV, best_betaH, best_R2))

    except Exception as e:
        print(f"[unpack/append error @ GHad={g:.3f} eV] {e}")
        continue  # keep going


# Summarize & save
if mechanism_choice == 0:
    df_vt = pd.DataFrame(results, columns=["GHad_eV", "k_V_best", "beta_V_best", "k_T_best", "R2_best"])
else:
    df_vt = pd.DataFrame(results, columns=["GHad_eV", "k_V_best", "beta_V_best", "beta_H_best", "R2_best"])

print("\nVolmer–Tafel optimization results:" if mechanism_choice == 0 else "\nVolmer–Heyrovsky optimization results:")
print(df_vt)
df_vt.to_excel("vt_bo_results.xlsx", index=False)
print("Saved: vt_bo_results.xlsx")
# ================== END VT BAYES OPT ==================

tafel_data = []  # store best-fit results for Tafel plotting

fig, ax = plt.subplots(figsize=(9, 7))

R2_MIN = 0.0  # only show fits with R² >= set value of R2_min

fig, ax = plt.subplots(figsize=(9, 7))

for (g, kV_best, betaV_best, param_best, r2_best) in results:
    if not np.isfinite(r2_best) or r2_best < R2_MIN:
        continue

    k_V = float(kV_best)
    if mechanism_choice == 0:   # VT
        k_T = float(param_best)
    else:                       # VH
        k_H = 1000.0 * k_V
        betaH_best = float(param_best)

    beta = [float(betaV_best), beta[1] if mechanism_choice == 0 else betaH_best]
    GHad_eV = float(g)
    GHad = GHad_eV * AvoNum * conversion_factor

    try:
        sol = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
        if (not sol.success) or (sol.y.shape[1] != len(t)):
            continue

        rates = np.array([rates_r0(tt, th) for tt, th in zip(t, sol.y.T)])
        I_model_full = rates[:, 0] * -F * 1000.0
        V_model_full = np.array([potential(tt) for tt in t])

        model_mask = (V_model_full >= FIT_MIN) & (V_model_full <= FIT_MAX)
        if model_mask.sum() < 8 or len(V_exp_masked) < 8:
            continue

        V_m = V_model_full[model_mask]
        I_m = I_model_full[model_mask]
        interp = interp1d(V_m, I_m, kind='linear', fill_value='extrapolate', assume_sorted=False)
        I_model_interp = interp(V_exp_masked.to_numpy())

        # RMSE
        eps = 1e-12
        y_true = np.log10(np.clip(np.abs(I_exp_masked.to_numpy()), eps, None))
        y_pred = np.log10(np.clip(np.abs(I_model_interp), eps, None))
        if len(y_true) > 8:
            y_true, y_pred = y_true[4:], y_pred[4:]
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))

        tafel_data.append({
            'GHad': g,
            'r2': float(r2_best),
            'rmse': rmse,
            'I_model': np.abs(I_model_interp.copy()),  # log-x wants positive
        })

        ax.plot(
            V_exp_masked, I_model_interp,
            label=f"GHad={g:.3f} eV | R²={float(r2_best):.3f} | RMSE={rmse:.3f}"
        )
    except Exception:
        continue

# experimental trace
ax.plot(V_exp_masked, I_exp_masked, 'k', lw=2.5, label=f"Experimental (scan {scan_id})")
ax.set_xlim(FIT_MIN, FIT_MAX)
ax.set_xlabel('Voltage vs. RHE (V)')
ax.set_ylabel('Kinetic current (mA/cm²)')
ax.set_title(f'{("Volmer–Tafel" if mechanism_choice==0 else "Volmer–Heyrovsky")} fits vs Experimental (scan {scan_id})')
ax.grid(True)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


fig_tafel, ax_tafel = plt.subplots(figsize=(9, 7))

# Plot stored model curves
for d in tafel_data:
    ax_tafel.plot(
        d['I_model'], V_exp_masked,
        label=f"GHad={d['GHad']:.3f} eV | R²={d['r2']:.3f} | RMSE={d['rmse']:.3f}"
    )

# Plot experimental (log scale, positive current)
ax_tafel.plot(np.abs(I_exp_masked), V_exp_masked, 'k', lw=2.5, label=f"Experimental (scan {scan_id})")

# Final formatting
ax_tafel.set_xscale('log')
ax_tafel.set_xlabel('Kinetic current (mA/cm²)')
ax_tafel.set_ylabel('Voltage vs. RHE (V)')
ax_tafel.set_title(f'Tafel Plot (scan {scan_id})')
ax_tafel.grid(True, which='both', ls='--')
ax_tafel.legend(fontsize=9)
fig_tafel.tight_layout()
plt.show()

