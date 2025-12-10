import numpy as np
import os
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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

######################################### FILE NAMING FUNCTION ############################################


def make_output_filename(kV, kT=None, kH=None, freq_array=None, beta=None,
                         dGmin=None, dGmax=None, voltage=None,
                         base="dynamic_simulation_output.xlsx"):
    """
    Build a unique filename string with parameters included.
    """
    # Format freq array nicely
    freq_str = "-".join([f"{f:.1e}" for f in (freq_array or [])])

    # Include kT only if relevant
    if kT is not None:
        k_str = f"kV={kV:.2e}_kT={kT:.2e}"
    elif kH is not None:
        k_str = f"kV={kV:.2e}_kH={kH:.2e}"
    else:
        k_str = f"kV={kV:.2e}"

    # Beta can be an array — include both
    if beta is not None:
        beta_str = "__".join([f"{b:.3f}" for b in beta])
    else:
        beta_str = "NA"

    # Build the base filename
    filename = (
        f"sim_k_{k_str}_freq_{freq_str}_beta_{beta_str}"
        f"_dG_{dGmin:.2f}-{dGmax:.2f}eV_V_{voltage:.2f}V.xlsx"
    )

    # Ensure uniqueness
    final_filename = filename
    counter = 1
    while os.path.exists(final_filename):
        final_filename = filename.replace(".xlsx", f"_{counter}.xlsx")
        counter += 1

    return final_filename


########################################  Time Function ###################################################


def make_t_eval(freq, t_switching=0.5, n_cycles=4,
                coarse_pts_per_period=40,
                halo_frac=0.05,
                halo_points=15):
    P = 2.0 / float(freq)
    t_end = t_switching + n_cycles * P

    dt = P / float(coarse_pts_per_period)
    # small epsilon to include t_end
    t_coarse = np.arange(0.0, t_end + 1e-12, dt)
    t_coarse = t_coarse[t_coarse <= t_end]        # hard clip

    # exact switch instants
    switches = t_switching + np.arange(n_cycles + 1) * P
    switches = np.clip(switches, 0.0, t_end)

    # halos around each switch
    hw = halo_frac * P
    halos_list = []
    for tk in switches:
        a = max(0.0, tk - hw)
        b = min(t_end, tk + hw)
        if b > a:
            halos_list.append(np.linspace(a, b, halo_points))
    halos = np.concatenate(
        halos_list) if halos_list else np.array([], dtype=float)

    t_eval = np.unique(np.concatenate([t_coarse, switches, halos]))
    t_eval = t_eval[(t_eval >= 0.0) & (t_eval <= t_end)]

    # Ensure exact endpoints are present
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
cmax = 7.5e-9     # sites/cm²
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = [0.443716, 0.1]
V_app = -0.2
tswitching = 0.5

k_V_RDS = 1e-9


freq_array = [1000, 100, 10, 1  ]

if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
    k_H = 0
elif mechanism_choice == 1:
    k_V = k_V_RDS
    k_H = k_V * 1000
    k_T = 0

# dG values, static volcano
dGmin_eV = -0.10 #eV
dGmax_eV = 0.20 #eV

# dG values, dynamic volcano
dGmin_dynamic = 0.05  # in eV
dGmax_dynamic = 0.15  # in eV


# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input(
    "Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input(
    "Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

# === Prepare dynamic overlay variables ===
dynamic_overlay_points = []
dynamic_overlay_by_freq = {}   # freq -> list of (GHad_eV, curr)
dynamic_overlay_by_freq1 = {}
dynamic_overlay_by_freq_rH = {}

avg_currents = []

r_T_list = []

# === DYNAMIC GHad(t) SIMULATION ===
if do_dynamic_ghad:
    T1_index = []
    T2_index = []
    print("\nRunning dynamic GHad(t) simulation...")
    thetaH_array = []
    # Time-varying GHad values (in J)
    dGmin = dGmin_dynamic * Avo * conversion_factor
    dGmax = dGmax_dynamic * Avo * conversion_factor

    # 1. Initialize lists to store the results per frequency
    avg_currents_dGmin = []
    avg_currents_dGmax = []
    dyn_results = []
    avg_rT_dGmin = []
    avg_rT_dGmax = []
    avg_rH_dGmin = []
    avg_rH_dGmax = []

    # Initial conditions
    # Initial coverage of Hads, needs to be high as this is reduction forward
    thetaA_H0_dynamic = 0.5
    thetaA_Star0_dynamic = 1.0 - thetaA_H0_dynamic  # Initial coverage of empty sites
    theta0_dynamic = [thetaA_Star0_dynamic, thetaA_H0_dynamic]

    for freq in freq_array:
        print(f"\nRunning simulation with period = {freq:.2e} Hz...")

        # time spacing
        t, max_time = make_t_eval(freq, t_switching=tswitching, n_cycles= 40,
                                  coarse_pts_per_period=60, halo_frac=0.1, halo_points=100)
        duration = [t[0], t[-1]]

        # keep the solver from skipping over switch neighborhoods
        P = 2.0 / freq
        maxstep = 2e-3

        print("Max Step: ", maxstep)

        duration = [0, max_time]
        time_index = [t]

        # function for defining how dGmin and dGmax are applied to the model
        def dGvt(t):
            if t < tswitching:
                return dGmin
            else:
                return dGmin if int((t - tswitching) * freq) % 2 == 0 else dGmax

        # setting potential for static hold
        def potential(t):
            return V_app

        # equil

        def eqpot(theta, GHad):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # unpack surface coverage

            # Volmer Tafel
            if mechanism_choice == 0:
                U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            # U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)
            U_H = 0
            if mechanism_choice == 1:
                U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
                U_H = (GHad / F) + (RT * np.log(thetaA_H / thetaA_star) / F)

            return U_V, U_H

        # reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this

        def rates_r0(t, theta):
            GHad = dGvt(t)
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # surface coverages again, acting as concentrations
            V = potential(t)  # Use t directly (scalar)
            # call function to find U for given theta
            U_V, U_H = eqpot(theta, GHad)

            # Volmer Rate Equation
            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * (
                np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

            r_T = 0
            if mechanism_choice == 0:
                T_1 = (thetaA_H ** 2)
                T_2 = (partialPH2 * (thetaA_star ** 2)
                       * np.exp((-2 * GHad) / RT))
                r_T = k_T * (T_1 - T_2)

                # r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))
            # Heyrovsky Rate Equation
            r_H = 0
            if mechanism_choice == 1:
                j1 = k_H * np.exp(-beta[1] * GHad / RT) * \
                    thetaA_star ** beta[1] * \
                    thetaA_H ** (1 - beta[1])
                exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
                exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                r_H = j1 * (exp21 - exp22)

            r_T_list.append(r_T)
            # T1_index.append(T_1)
            # T2_index.append(T_2)
            return r_V, r_T, r_H

        def sitebal(t, theta):
            r_V, r_T, r_H = rates_r0(t, theta)
            if mechanism_choice in [0]:
                thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
                thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
                # [0 = star, 1 = H]
                dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT]
            elif mechanism_choice in [1]:
                # summing all step rates based on how they affect theta_star
                theta_star_rate = r_H - r_V
                theta_H_rate = r_V - r_H  # summing all step rates based on how they affect theta_H
                dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
            return dthetadt

        soln = solve_ivp(sitebal, duration, theta0_dynamic,
                         t_eval=t, max_step=maxstep, method='BDF')
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

        avg_rT_vals = np.average(r_T_vals)
        print(f"Average rT at {freq}:", avg_rT_vals)

        # Mask for each dynamic GHad value
        mask_min = (t >= tswitching) & np.isclose(
            GHad_t_J, dGmin, rtol=1e-8, atol=1e-20)
        mask_max = (t >= tswitching) & np.isclose(
            GHad_t_J, dGmax, rtol=1e-8, atol=1e-20)

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

        # Absolute value of the minimum current (max magnitude)
        avg_curr_at_dGmin = np.average(np.abs(curr_dynamic[mask_min]))
        avg_curr_at_dGmax = np.average(np.abs(curr_dynamic[mask_max]))

        avg_currents_dGmin.append(avg_curr_at_dGmin)
        avg_currents_dGmax.append(avg_curr_at_dGmax)

        # Save them for overlay plotting
        dynamic_overlay_points.append((dGmin_dynamic, avg_curr_at_dGmin))
        dynamic_overlay_points.append((dGmax_dynamic, avg_curr_at_dGmax))

        # ... inside: for freq in freq_array:
        # Save per-freq overlay points (in eV for the x-axis of the volcano plot)
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
        
# =============================================================================
#         print(f"Max |Current| at GHad = {dGmin_dynamic:.2f} eV: {avg_curr_at_dGmin:.3f} mA/cm²")
#         print(f"Max |Current| at GHad = {dGmax_dynamic:.2f} eV: {avg_curr_at_dGmax:.3f} mA/cm²")
# =============================================================================

        dyn_results.append({
            "r_T": r_T_vals,
            "r_H": r_H_vals,
            "rV": r_V_vals,
            "period": 2 / (freq),
            "freq": float(freq),
            "t": t.copy(),
            "curr": curr_dynamic.copy(),
            "thetaH": thetaH_array.copy(),
            "GHad_eV": GHad_t_eV.copy(),
            "Average Current": avg_curr,
            # "curr_at_dGmin": avg_curr_at_dGmin,
            # "curr_at_dGmax": avg_curr_at_dGmax,
        })

    freq_labels = [f"{res['freq']:.2e}" for res in dyn_results]
# =============================================================================
#     curr_at_dGmin = [res["curr_at_dGmin"] for res in dyn_results]
#     curr_at_dGmax = [res["curr_at_dGmax"] for res in dyn_results]
# =============================================================================

# =============================================================================
#     cycles = 6
# 
#     fig, ax1 = plt.subplots(figsize=(8, 5))
# 
#     # Left axis (rT values)
#     ax1.scatter(freq_labels, avg_rT_dGmin, color="blue", marker="o",
#                 label=f"Avg rT at {dGmin_dynamic:.2f} eV")
#     ax1.scatter(freq_labels, avg_rT_dGmax, color="blue", marker="o",
#                 label=f"Avg rT at {dGmax_dynamic:.2f} eV")
#     ax1.set_ylabel("TOF (mol/cm²·s)", color="black")
#     ax1.tick_params(axis='y', labelcolor="black")
# 
#     # Right axis (Current values)
#     ax2 = ax1.twinx()
#     ax2.scatter(freq_labels, avg_currents_dGmin, color="red", marker="x",
#                 label=f"Avg |Current| at {dGmin_dynamic:.2f} eV")
#     ax2.scatter(freq_labels, avg_currents_dGmax, color="red", marker="x",
#                 label=f"Avg |Current| at {dGmax_dynamic:.2f} eV")
#     ax2.set_ylabel("Current Density (mA/cm²)", color="black")
#     ax2.tick_params(axis='y', labelcolor="black")
# 
#     # Shared X-axis
#     ax1.set_xlabel("Frequency (Hz)")
#     ax1.set_title("Dynamic GHad(t): Average rT and Current vs Frequency")
#     ax1.grid(True)
# 
#     # Combine legends from both axes
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
# 
#     plt.tight_layout()
#     plt.show()
# =============================================================================

    # === Excel Export: One sheet per frequency ===
    output_filename = make_output_filename(
        kV=k_V,
        kT=k_T if mechanism_choice == 0 else None,
        kH=k_H if mechanism_choice == 1 else None,
        freq_array=freq_array,
        beta=beta,
        dGmin=dGmin_dynamic,
        dGmax=dGmax_dynamic,
        voltage=potential(0)
    )

for res in dyn_results:
    freq = res["freq"]
    t = res["t"]
    rT_vals = res["r_T"]
    rH_vals = res["r_H"]

    # period for this frequency
    P = 2.0 / freq

    # how many cycles to display
    cycles_to_plot = 20

    # static hold (same as in your simulation)
    t_switching = tswitching

    # define time window
    t_start = t_switching
    t_end = t_switching + cycles_to_plot * P

    mask = (t >= t_start) & (t <= t_end)
    
    if mechanism_choice == 0:
    # make separate plot for each frequency
        plt.figure(figsize=(8, 5))
        plt.plot(t[mask], rT_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel(r"$r_T$ (mol/cm²·s)")
        plt.title(f"r_T vs Time at {freq:.2e} Hz (first {cycles_to_plot} cycles)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    if mechanism_choice == 1:
        plt.figure(figsize=(8, 5))
        plt.plot(t[mask], rH_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel(r"$r_H$ (mol/cm²·s)")
        plt.title(f"r_H vs Time at {freq:.2e} Hz (first {cycles_to_plot} cycles)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# === STATIC VOLCANO PLOT ===
if do_static_volcano:
    print("\nRunning static volcano plot...")

    # Initial conditions
    thetaA_H0_static = 0.5
    thetaA_Star0_static = 1.0 - thetaA_H0_static
    theta0_static = [thetaA_Star0_static, thetaA_H0_static]

    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 12)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor
    GHad_results = []
    GHad_results1 = []
    GHad_results_rH = []
    static_rT_dict = {}
    avg_summary = []

    t_end = 200
    t_eval = np.linspace(0, t_end, 1000)

    for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):

        def potential_static(t):
            return V_app

        def eqpot_static(theta):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta
            U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            U_H = 0
            if mechanism_choice == 1:
                U_11 = GHad / F
                U_12 = (RT / F) * np.log(thetaA_H / thetaA_star)
                U_H = U_11 + U_12
            return U_V, U_H

        def rates_r0_static(t_static, theta):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta
            V = potential_static(t_static)
            U_V, U_H = eqpot_static(theta)

            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
                * np.exp(beta[0] * GHad / RT) * (
                    np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

            r_T = 0
            if mechanism_choice == 0:
                r_T = k_T * ((thetaA_H ** 2) - (partialPH2 *
                                                (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))

            r_H = 0
            if mechanism_choice == 1:
                j1 = k_H * np.exp(-beta[1] * GHad / RT) \
                    * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
                exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                r_H = j1 * (exp21 - exp22)

            return r_V, r_T, r_H

        def sitebal_static(t_static, theta):
            r_V, r_T, r_H = rates_r0_static(t_static, theta)
            if mechanism_choice in [0]:
                thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
                thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
                dthetadt = [thetaStar_rate_VT, thetaH_rate_VT]
            elif mechanism_choice in [1]:
                theta_star_rate = r_H - r_V
                theta_H_rate = r_V - r_H
                dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
            return dthetadt

        soln = solve_ivp(sitebal_static, (0, t_end), theta0_static,
                         t_eval=t_eval, method='BDF', max_step=0.10)
        r0_vals = np.array([rates_r0_static(time, theta)
                           for time, theta in zip(t_eval, soln.y.T)])
        curr_static = r0_vals[:, 0] * -F * 1000  # mA/cm²
        average_current = np.abs(np.average(curr_static))
        rT_vals = r0_vals[:, 1]
        rH_vals = r0_vals[:, 2]
        averagerT = np.average(rT_vals)
        average_rH = np.average(rH_vals)
        
        if mechanism_choice == 0:
            print(f"Average rT for {GHad_eV}:", averagerT)
        if mechanism_choice == 1:
            print(f"Average rH for {GHad_eV}:", average_rH)

        # Store only summary into volcano plot
        GHad_results.append((GHad_eV, averagerT))
        GHad_results1.append((GHad_eV, average_current))
        GHad_results_rH.append((GHad_eV, average_rH))
        static_rT_dict[f"{GHad_eV:.3f} eV"] = rT_vals
        avg_summary.append({
            "GHad (eV)": GHad_eV,
            "Average r_T (mol/cm²·s)": averagerT,
            "Average Current (mA/cm²)": average_current
        })

    if mechanism_choice == 0:
        
            # Volcano plot (rT)
        GHad_vals, abs_rT = zip(*GHad_results)
        plt.figure(figsize=(10, 8))
        plt.plot(GHad_vals, abs_rT, label='Static GHad Scan', marker='o', color="blue")
        
            
        if dynamic_overlay_by_freq:
            freq_markers = {freq_array[0]: "D", freq_array[1]: "^"}
            freq_colors = {freq_array[0]: "red", freq_array[1]: "tab:orange"}
        
            for f in freq_array:
                if f not in dynamic_overlay_by_freq:
                    continue
                pts = dynamic_overlay_by_freq[f]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
        
                # scatter markers
                plt.scatter(xs, ys,
                            marker=freq_markers.get(f, "x"),
                            s=120,
                            facecolors="none",
                            edgecolors=freq_colors.get(f, None),
                            linewidths=2,
                            zorder=3)
        
                # inline text labels
                for x, y in zip(xs, ys):
                    plt.text(
                        x, y,
                        f"{int(round(f))} Hz",  # no decimals
                        fontsize=16, weight="bold",
                        color=freq_colors.get(f, "black"),
                        ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
                    )
        
        plt.xlabel("GHad (eV)")
        plt.ylabel("Average $\mathbf{r_T}$ (mol/cm²·s)")
        plt.title(f"Volcano Plot: Avg $\mathbf{{r_T}}$ vs GHad, V = {V_app} V")
        #plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        plt.tight_layout()
        plt.show()
        
    if mechanism_choice == 1:
    
        # Volcano plot (rH)
        GHad_vals, avg_rH = zip(*GHad_results_rH)
        plt.figure(figsize=(10, 8))
        plt.plot(GHad_vals, avg_rH, label='Static GHad Scan', marker='o', color="blue")
    
        if dynamic_overlay_by_freq_rH:
            freq_markers = {freq_array[0]: "D", freq_array[1]: "^"}
            freq_colors = {freq_array[0]: "red", freq_array[1]: "tab:orange"}
        
            for f in freq_array:
                if f not in dynamic_overlay_by_freq_rH:
                    continue
                pts = dynamic_overlay_by_freq_rH[f]
                xs = [p[0] for p in pts]  # GHad (eV)
                ys = [p[1] for p in pts]  # Current
        
                # scatter markers
                plt.scatter(xs, ys,
                            marker=freq_markers.get(f, "x"),
                            s=120,
                            facecolors="none",
                            edgecolors=freq_colors.get(f, None),
                            linewidths=2,
                            zorder=3)
        
                # inline text labels
                for x, y in zip(xs, ys):
                    plt.text(
                        x, y,
                        f"{int(round(f))} Hz",  # integer only
                        fontsize=10, weight="bold",
                        color=freq_colors.get(f, "black"),
                        ha="left", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
                    )
        
        plt.xlabel("GHad (eV)")
        plt.ylabel("Max rH (mol/cm² s)")
        plt.title(f"Volcano Plot: Max rH vs GHad, V = {V_app} V")
        #plt.grid(True)
        plt.tight_layout()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        plt.show()

    # Volcano plot (Current)
    GHad_vals, abs_currents = zip(*GHad_results1)
    plt.figure(figsize=(10, 8))
    plt.plot(GHad_vals, abs_currents, label='Static GHad Scan', marker='o', color="blue")
    
    if dynamic_overlay_by_freq1:
        freq_markers = {freq_array[0]: "D", freq_array[1]: "^"}
        freq_colors = {freq_array[0]: "red", freq_array[1]: "tab:orange"}
    
        for f in freq_array:
            if f not in dynamic_overlay_by_freq1:
                continue
            pts = dynamic_overlay_by_freq1[f]
            xs = [p[0] for p in pts]  # GHad (eV)
            ys = [p[1] for p in pts]  # Current
    
            # scatter markers
            plt.scatter(xs, ys,
                        marker=freq_markers.get(f, "x"),
                        s=120,
                        facecolors="none",
                        edgecolors=freq_colors.get(f, None),
                        linewidths=2,
                        zorder=3)
    
            # inline text labels
            for x, y in zip(xs, ys):
                plt.text(
                    x, y,
                    f"{int(round(f))} Hz",  # integer only
                    fontsize=10, weight="bold",
                    color=freq_colors.get(f, "black"),
                    ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
                )
    
    plt.xlabel("GHad (eV)")
    plt.ylabel("Max |Current Density| (mA/cm²)")
    plt.title(f"Volcano Plot: Max Current vs GHad, V = {V_app} V")
    #plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    plt.show()


    max_index = np.argmax(abs_currents)
    print(
        f"\nMax Current (static): {abs_currents[max_index]:.3f} mA/cm² at GHad = {GHad_vals[max_index]:.3f} eV")
    print("\nStatic Volcano Summary:")
    for g, c in GHad_results:
        print(f"GHad = {g:.3f} eV → Average r_T = {c:.3e} mol/cm²·s")

# ======================== EXCEL EXPORT (ALL TOGETHER) ========================
save_folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
output_filename = os.path.join(
    save_folder,
    make_output_filename(kV=k_V, kT=k_T, kH=k_H,
                         freq_array=freq_array, beta=beta,
                         dGmin=dGmin_dynamic, dGmax=dGmax_dynamic,
                         voltage=V_app)
)

with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
    # Dynamic sheets
    for res in dyn_results:
        freq_label = f"{res['freq']:.2e}Hz"
        df = pd.DataFrame({
            "Time (s)": res["t"],
            "GHad (eV)": res["GHad_eV"],
            "Current (mA/cm²)": res["curr"],
            "r_T (mol/cm²·s)": res["r_T"],
            "r_V (mol/cm²·s)": res["rV"],
            "θ_H": res["thetaH"]
        })
        df.to_excel(writer, sheet_name=freq_label[:30], index=False)

    # Static sheets (each GHad)
    for label, rT_vals in static_rT_dict.items():
        df = pd.DataFrame({"Time (s)": t_eval, "r_T (mol/cm²·s)": rT_vals})
        df.to_excel(writer, sheet_name=f"GHad_{label}"[:31], index=False)

    # All static in one sheet
    if static_rT_dict:
        df_static_all = pd.DataFrame(static_rT_dict)
        df_static_all.insert(0, "Time (s)", t_eval)
        df_static_all.to_excel(writer, sheet_name="All_Static", index=False)

    # Summary sheet
    if avg_summary:
        df_summary = pd.DataFrame(avg_summary)
        df_summary.to_excel(writer, sheet_name="Static_Summary", index=False)

print(f"\nAll results exported to Excel: {output_filename}")