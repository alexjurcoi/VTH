import numpy as np
import os
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

plt.rcParams.update({
    "font.size": 20,        # increase overall font size
    "axes.labelsize": 20,   # bigger axis labels
    "axes.labelweight": "bold",
    "axes.titlesize": 24,   # bigger title
    "axes.titleweight": "bold",
    "legend.fontsize": 16,  # legend text size
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

###############################################################################
# MECHANISM MODE SELECTION (0=VT, 1=VH, 2=BOTH)
###############################################################################

while True:
    mech_mode = input(
        "Run which mechanism(s)?\n"
        f"{'Volmer RDS Tafel Fast (VT only) [0]'.rjust(40)}\n"
        f"{'Volmer RDS Heyrovsky Fast (VH only) [1]'.rjust(40)}\n"
        f"{'Both VT and VH [2]'.rjust(40)}\n"
        "Enter 0, 1, or 2: "
    ).strip()
    if mech_mode in ["0", "1", "2"]:
        mech_mode = int(mech_mode)
        break
    print("Invalid choice. Please enter 0, 1, or 2.")

MECH_LABEL = {0: "VT", 1: "VH"}
if mech_mode == 0:
    mechanisms_to_run = [0]       # VT only
elif mech_mode == 1:
    mechanisms_to_run = [1]       # VH only
else:
    mechanisms_to_run = [0, 1]    # BOTH

###############################################################################
# FILE NAMING FUNCTION (kV, kT, kH always in filename)
###############################################################################

def make_output_filename(kV, kT, kH, freq_array=None, beta=None,
                         dGmin=None, dGmax=None, voltage=None,
                         
                         base="dynamic_simulation_output.xlsx"):
    """
    Build a unique filename string with parameters included.
    kV, kT, kH always appear in the filename, regardless of mechanism.
    """
    freq_str = "-".join([f"{f:.1e}" for f in (freq_array or [])])
    k_str = f"kV={kV:.2e}_kT={kT:.2e}_kH={kH:.2e}"

    if beta is not None:
        beta_str = "__".join([f"{b:.3f}" for b in beta])
    else:
        beta_str = "NA"

    filename = (
        f"sim_k_{k_str}_freq_{freq_str}_beta_{beta_str}"
        f"_dG_{dGmin:.2f}-{dGmax:.2f}eV_V_{voltage:.2f}.xlsx"
    )

    final_filename = filename
    counter = 1
    while os.path.exists(final_filename):
        final_filename = filename.replace(".xlsx", f"_{counter}.xlsx")
        counter += 1
    return final_filename

########################################  Time Function ###################################################

def make_t_eval(freq, n_cycles=20, pts_per_freq = 100):
    P = 1.0 / float(freq)
    t_end = n_cycles * P
    dt = P / pts_per_freq
    t_eval = np.arange(0, t_end + dt, dt) 
    t_eval = t_eval[(t_eval >= 0.0) & (t_eval <= t_end)]

    # failsafe
    if t_eval.size == 0 or t_eval[0] > 0.0:
        t_eval = np.insert(t_eval, 0, 0.0)
    if t_eval[-1] < t_end:
        t_eval = np.append(t_eval, t_end)

    return t_eval, t_end

###########################################################################################################################
###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################
###########################################################################################################################

RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-9 * 10     # sites/cm²
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23     # 1/mol
partialPH2 = 1.0
beta = [0.5, 0.5]
V_app = -0.1
k = 95
a_tol = 1e-14
r_tol = 1e-8
phi = -np.pi/2
duty = 0.5      #0.5 is 50/50 split, 0.1 -> majority dGmin | 0.9 -> majority dGmax

k_V_RDS = 1e-10

#   kT and kH (for naming + magnitude)
k_T_base = k_V_RDS * 20000
k_H_base = k_V_RDS * 500

freq_norm_array = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])
#freq_norm_array = np.array([1e-3])
freq_array = freq_norm_array / (k_V_RDS / cmax)

# dG values, static volcano
dGmin_eV = -0.1  # eV
dGmax_eV = 0.50   # eV

# dG values, dynamic volcano
dGmin_dynamic = 0.10  # eV
dGmax_dynamic = 0.30  # eV

# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

###############################################################################
# STORAGE FOR ALL MECHANISMS
###############################################################################

avg_rV_freq = []
avg_rT_freq = []
avg_rH_freq = []

# Dynamic results per mechanism (for Excel)
dyn_results_all = {"VT": [], "VH": []}

# Dynamic overlay storage per mechanism (for volcano plots)
overlay_storage = {
    "VT": {"rT": {}, "rH": {}, "curr": {}, "avg": {}},
    "VH": {"rT": {}, "rH": {}, "curr": {}, "avg": {}},
}

# For static volcano summary
static_summary_rows = []  # will hold dicts with mechanism, GHad, avg_rT/r_H, current

# Remember last maxstep per mechanism for titles
last_maxstep_per_mech = {"VT": None, "VH": None}

def plot_block_heatmap(Z, dG_vals, title="", cbar_label=""):
    step = float(np.round(dG_vals[1] - dG_vals[0], 10))
    edges = np.concatenate(([dG_vals[0] - step/2],
                            (dG_vals[:-1] + dG_vals[1:]) / 2,
                            [dG_vals[-1] + step/2]))
    X, Y = np.meshgrid(edges, edges)

    fig, ax = plt.subplots(figsize=(9, 8))
    m = ax.pcolormesh(X, Y, Z, shading="flat")  # crisp blocks

    # diagonal boundary dGmin=dGmax
    ax.plot([dG_vals[0], dG_vals[-1]], [dG_vals[0], dG_vals[-1]], "--", linewidth=2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dGmin (eV)", fontweight="bold")
    ax.set_ylabel("dGmax (eV)", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    cb = fig.colorbar(m, ax=ax)
    cb.set_label(cbar_label, fontweight="bold")
    plt.tight_layout()
    plt.show()


###############################################################################
# === DYNAMIC GHad(t) SIMULATION ===
###############################################################################

if do_dynamic_ghad:

    print("\nRunning dynamic GHad(t) simulation...")

    # =========================
    # dGmin/dGmax sweep settings
    # =========================
    dG_lo, dG_hi, dG_step = -0.10, 0.40, 0.05
    dG_vals = np.round(np.arange(dG_lo, dG_hi + 1e-12, dG_step), 2)
    n_dG = len(dG_vals)
    total_iterations = ((n_dG * (n_dG + 1)) // 2 * len(freq_norm_array))
    
    # choose what scalar to color the heatmap with:
    #   "avg_curr_metric"  -> 0.5*(avg|curr| near dGmin + near dGmax)
    #   "avg_curr_abs"  -> abs(average over entire waveform)
    #   "avg_rT_metric"    -> 0.5*(avg rT near dGmin + near dGmax)
    #   "avg_rH_metric"    -> 0.5*(avg rH near dGmin + near dGmax)
    heatmap_metric_curr = "avg_curr_metric"
    heatmap_metric_VT = "avg_rT_metric"
    heatmap_metric_VH = "avg_rH_metric"
    
    # heatmaps[mech_label][freq] = Z matrix
    heatmaps_curr = {"VT": {}, "VH": {}}
    heatmaps_rate = {"VT": {}, "VH": {}}
    
    # Time-varying GHad values (in J)
    dGmin = dGmin_dynamic * Avo * conversion_factor
    dGmax = dGmax_dynamic * Avo * conversion_factor

    for mechanism_choice in mechanisms_to_run:
        mech_label = MECH_LABEL[mechanism_choice]
        if mech_label == "VT":
            heatmap_metric_rate = heatmap_metric_VT   # rT metric
        else:
            heatmap_metric_rate = heatmap_metric_VH   # rH metric
        print(f"\n=== DYNAMIC SIMULATION FOR {mech_label} ===")

        # Set mechanism-specific rate constants (Option 1)
        if mechanism_choice == 0:   # VT
            k_V = k_V_RDS
            k_T = k_T_base
            k_H = 0.0
        else:                       # VH
            k_V = k_V_RDS
            k_T = 0.0
            k_H = k_H_base

        # These match your original script naming / structure
        dynamic_overlay_points = []
        dynamic_overlay_by_freq = {}    # r_T overlay
        dynamic_overlay_by_freq1 = {}   # current overlay
        dynamic_overlay_by_freq_rH = {} # r_H overlay

        avg_currents = []

        # Lists per frequency
        avg_currents_dGmin = []
        avg_currents_dGmax = []
        avg_rT_dGmin = []
        avg_rT_dGmax = []
        avg_rH_dGmin = []
        avg_rH_dGmax = []

        dyn_results = []
        count = 0
        
        for freq in freq_array:

            print(f"\n=== HEATMAP SWEEP @ freq = {freq:.2e} Hz ({mech_label}) ===")
        
            # Z rows = dGmax index, Z cols = dGmin index
            Z_curr = np.full((n_dG, n_dG), np.nan, dtype=float)
            Z_rate = np.full((n_dG, n_dG), np.nan, dtype=float)
        
            # dGmax: 0.40 -> -0.10 (descending)
            for i_max in range(n_dG - 1, -1, -1):
                dGmax_dynamic = float(dG_vals[i_max])
                dGmax = dGmax_dynamic * Avo * conversion_factor
        
                # dGmin: -0.10 -> dGmax (ascending)
                for i_min in range(0, i_max + 1):
                    dGmin_dynamic = float(dG_vals[i_min])
                    dGmin = dGmin_dynamic * Avo * conversion_factor
                    
                    count += 1
                    ###### printing count number ################
                    print(f"Iteration {count} of {total_iterations}")
                    
                    try:
                        # time spacing
                        t, max_time = make_t_eval(freq, n_cycles=100, pts_per_freq=200)            
                        duration = [0, max_time]
            
                        # keep the solver from skipping over switch neighborhoods
                        P = 1.0 / freq
                        maxstep = 1 / (freq * 1e3)
            
                        def dGvt(t):
                            return (dGmin) + (dGmax - dGmin) * (np.tanh(k * (np.sin(2*np.pi * freq * t - phi) - np.sin(np.pi * (0.5 - duty)))) + 1) / 2
            
                        # static potential
                        def potential(t):
                            return V_app
            
                        # equilibrium potentials
                        def eqpot(theta, GHad):
                            theta = np.asarray(theta)
                            thetaA_star, thetaA_H = theta  # unpack surface coverage
            
                            U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
                            U_H = 0.0
                            if mechanism_choice == 1:  # VH
                                U_H = (GHad / F) + (RT * np.log(thetaA_H / thetaA_star) / F)
                            return U_V, U_H
            
                        # reduction is FORWARD, oxidation is REVERSE
                        def rates_r0(t, theta):
                            GHad = dGvt(t)
                            theta = np.asarray(theta)
                            thetaA_star, thetaA_H = theta
                            V = potential(t)
                            U_V, U_H = eqpot(theta, GHad)
                            
                            if (thetaA_star <= 0) or (thetaA_H <= 0) or (thetaA_star >= 1) or (thetaA_H >= 1) or (not np.isfinite(thetaA_star)) or (not np.isfinite(thetaA_H)):
                                raise RuntimeError(f"Bad theta at t={t:.3e}: theta*={thetaA_star}, thetaH={thetaA_H}")
            
                            # Volmer Rate Equation
                            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
                                * np.exp(beta[0] * GHad / RT) * (
                                    np.exp(-(beta[0]) * F * (V - U_V) / RT)
                                    - np.exp((1 - beta[0]) * F * (V - U_V) / RT))
            
                            r_T = 0.0
                            if mechanism_choice == 0:  # VT
                                T_1 = (thetaA_H ** 2)
                                T_2 = (partialPH2 * (thetaA_star ** 2)
                                       * np.exp((-2 * GHad) / RT))
                                r_T = k_T * (T_1 - T_2)
            
                            r_H = 0.0
                            if mechanism_choice == 1:  # VH
                                j1 = k_H * np.exp(-beta[1] * GHad / RT) * \
                                    thetaA_star ** beta[1] * \
                                    thetaA_H ** (1 - beta[1])
                                exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
                                exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                                r_H = j1 * (exp21 - exp22)
            
                            return r_V, r_T, r_H
            
                        def theta_H_eq_dynamic(GHad_init, V, mech_choice):
                            # bracket for theta, solution should be between 1e-9 and 1 - 1e-9
                            lo, hi = 1e-9, 1 - 1e-9
            
                            def f(thetaH):
                                theta = np.array([1 - thetaH, thetaH])
                                rV, rT, rH = rates_r0(0, theta)
                                if mech_choice == 0:
                                    return rV - 2 * rT
                                else:
                                    return rV - rH
            
                            sol = root_scalar(f, bracket=[lo, hi])
                            return sol.root
                                    
                        # Initial coverage of Hads, inside loop so that it starts fresh each time
                        thetaA_H0_dynamic = theta_H_eq_dynamic(dGvt(0), V_app, mechanism_choice)
                        thetaA_Star0_dynamic = 1.0 - thetaA_H0_dynamic  # Initial coverage of empty sites
                        theta0_dynamic = [thetaA_Star0_dynamic, thetaA_H0_dynamic]
            
                        def sitebal(t, theta):
                            r_V, r_T, r_H = rates_r0(t, theta)
                            if mechanism_choice == 0:
                                thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
                                thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
                                dthetadt = [thetaStar_rate_VT, thetaH_rate_VT]
                            else:
                                theta_star_rate = r_H - r_V
                                theta_H_rate = r_V - r_H
                                dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
                            return dthetadt
            
                        soln = solve_ivp(sitebal, duration, theta0_dynamic,
                                         t_eval=t, max_step=maxstep, method='BDF', atol = a_tol, rtol = r_tol)
                        theta_at_t = soln.y  # shape: (2, len(t))
                        thetaH_array = theta_at_t[1, :]
            
                        GHad_t_J = np.array([dGvt(time) for time in t])
                        GHad_t_eV = GHad_t_J / (Avo * conversion_factor)
            
                        r0_vals = np.array([rates_r0(time, theta)
                                            for time, theta in zip(t, theta_at_t.T)])
                        r_V_vals = r0_vals[:, 0]
                        r_T_vals = r0_vals[:, 1]
                        r_H_vals = r0_vals[:, 2]
            
                        curr_dynamic = r_V_vals * -F * 1000  # mA/cm²
            
                        avg_curr = np.abs(np.average(curr_dynamic))
                        avg_currents.append(avg_curr)
                        
                        average_rT = np.average(r_T_vals)
                        average_rH = np.average(r_H_vals)
            
                        GHad_range = dGmax - dGmin
                        mask_min = (np.abs(GHad_t_J - dGmin) < 0.2 * GHad_range)
                        mask_max = (np.abs(GHad_t_J - dGmax) < 0.2 * GHad_range)
            
                        average_rT_dGmin = np.average(r_T_vals[mask_min])
                        average_rT_dGmax = np.average(r_T_vals[mask_max])
                        average_rH_dGmin = np.average(r_H_vals[mask_min])
                        average_rH_dGmax = np.average(r_H_vals[mask_max])
            
                        print(f"Average rT at {dGmin}:", average_rT_dGmin)
                        print(f"Average rT at {dGmax}:", average_rT_dGmax)
                        print(f"Average rH at {dGmin}:", average_rH_dGmin)
                        print(f"Average rH at {dGmax}:", average_rH_dGmax)
            
                        avg_rT_dGmin.append(average_rT_dGmin)
                        avg_rT_dGmax.append(average_rT_dGmax)
                        avg_rH_dGmin.append(average_rH_dGmin)
                        avg_rH_dGmax.append(average_rH_dGmax)
            
                        # Absolute value of the current at GHad min/max
                        avg_curr_at_dGmin = np.average(np.abs(curr_dynamic[mask_min]))
                        avg_curr_at_dGmax = np.average(np.abs(curr_dynamic[mask_max]))
                        avg_currents_dGmin.append(avg_curr_at_dGmin)
                        avg_currents_dGmax.append(avg_curr_at_dGmax)
            
                        # Save them for overlay plotting (per-frequency)
                        dynamic_overlay_points.append((dGmin_dynamic, avg_curr_at_dGmin))
                        dynamic_overlay_points.append((dGmax_dynamic, avg_curr_at_dGmax))
            
                        dynamic_overlay_by_freq[freq] = [
                            (dGmin_dynamic, float(average_rT_dGmin)),
                            (dGmax_dynamic, float(average_rT_dGmax)),
                        ]
                        dynamic_overlay_by_freq1[freq] = [
                            (dGmin_dynamic, float(avg_curr_at_dGmin)),
                            (dGmax_dynamic, float(avg_curr_at_dGmax)),
                        ]
                        dynamic_overlay_by_freq_rH[freq] = [
                            (dGmin_dynamic, float(average_rH_dGmin)),
                            (dGmax_dynamic, float(average_rH_dGmax)),
                        ]
            
                        dyn_results.append({
                            "r_T": r_T_vals,
                            "r_H": r_H_vals,
                            "rV": r_V_vals,
                            "period": 1 / (freq),
                            "freq": float(freq),
                            "t": t.copy(),
                            "curr": curr_dynamic.copy(),
                            "thetaH": thetaH_array.copy(),
                            "GHad_eV": GHad_t_eV.copy(),
                            "Average Current": avg_curr,
                            "maxstep": maxstep,
                        })
                        
                        # choose what to store
                        if heatmap_metric_curr == "avg_curr_metric":
                            Z_curr[i_max, i_min] = avg_curr
                        else:
                            raise ValueError("Unknown heatmap_metric_curr")
                        
                        if heatmap_metric_rate == "avg_rT_metric":
                            Z_rate[i_max, i_min] = average_rT
                        elif heatmap_metric_rate == "avg_rH_metric":
                            Z_rate[i_max, i_min] = average_rH
                        else:
                            raise ValueError("Unknown heatmap_metric")
                                    
                        # OPTIONAL time-domain plots (same as your original; keep or comment out)
                        P = 1.0 / freq
                        cycles_to_plot_binding = 5
                        cycles_to_plot_coverage = 5
                        t_start = 0
                        t_end_plot = cycles_to_plot_coverage * P
                        t_end_binding = cycles_to_plot_binding * P
                        mask = (t >= t_start) & (t <= t_end_plot)
                        mask_binding = (t >= t_start) & (t <= t_end_binding)
                    
                    except Exception as e:
                        print(f"FAIL freq={freq:.2e}, dGmin={dGmin_dynamic:.2f}, dGmax={dGmax_dynamic:.2f}: {e}")
                        Z_curr[i_max, i_min] = np.nan
                        Z_rate[i_max, i_min] = np.nan

            # store Z for this frequency
            heatmaps_curr[mech_label][float(freq)] = Z_curr
            heatmaps_rate[mech_label][float(freq)] = Z_rate

            # store last maxstep for this mechanism
            last_maxstep_per_mech[mech_label] = maxstep
        
            overlay_storage[mech_label]["avg"][freq] = {
                "average_curr": avg_curr,
                "average_rT": average_rT,
                "average_rH": average_rH,
            }

        # save dynamic results & overlays for this mechanism
        dyn_results_all[mech_label] = dyn_results
        overlay_storage[mech_label]["rT"] = dynamic_overlay_by_freq
        overlay_storage[mech_label]["rH"] = dynamic_overlay_by_freq_rH
        overlay_storage[mech_label]["curr"] = dynamic_overlay_by_freq1
      

    ################### Heatmap Plotting #############################
    
    #current heatmap
    for mechanism_choice in mechanisms_to_run:
        mech_label = MECH_LABEL[mechanism_choice]
        for freq in freq_array:
            Z = heatmaps_curr[mech_label][float(freq)]
            plot_block_heatmap(
                Z, dG_vals,
                title=f"{mech_label} {heatmap_metric_curr},\n Duty={duty} Voltage = {V_app} @ {freq:.2e} Hz",
                cbar_label=heatmap_metric_curr
            )
    #rate heatmap
    for mechanism_choice in mechanisms_to_run:
        mech_label = MECH_LABEL[mechanism_choice]
        for freq in freq_array:
            Z = heatmaps_rate[mech_label][float(freq)]
            plot_block_heatmap(
                Z, dG_vals,
                title=f"{mech_label} {heatmap_metric_rate},\n Duty={duty} Voltage = {V_app} @ {freq:.2e} Hz",
                cbar_label=heatmap_metric_rate
            )

# ======================== EXCEL EXPORT (ALL TOGETHER) ========================
save_folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
flattened_data = []

for mech_label, freq_dict in heatmaps_curr.items():
    for freq, Z_matrix in freq_dict.items():
        for i_max in range(n_dG):
            for i_min in range(n_dG):
                val = Z_matrix[i_max, i_min]
                
                # Only export if it's not a NaN (the empty half of the triangle)
                if not np.isnan(val):
                    flattened_data.append({
                        "Mechanism": mech_label,
                        "Frequency_Hz": freq,
                        "dG_max_eV": dG_vals[i_max],
                        "dG_min_eV": dG_vals[i_min],
                        "Metric_Value": val
                    })

# Create one master dataframe
df_master = pd.DataFrame(flattened_data)
df_master.to_excel("Dynamic_Simulation_Results.xlsx", index=False)