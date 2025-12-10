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
                         t_switching=None,
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
        f"_dG_{dGmin:.2f}-{dGmax:.2f}eV_V_{voltage:.2f}_tswitching{t_switching}.xlsx"
    )

    final_filename = filename
    counter = 1
    while os.path.exists(final_filename):
        final_filename = filename.replace(".xlsx", f"_{counter}.xlsx")
        counter += 1
    return final_filename

########################################  Time Function ###################################################

def make_t_eval(freq, n_cycles=20,
                coarse_pts_per_period=4,
                halo_frac=0.05, max_eq_pts=500,
                halo_points=40):
    P = 2.0 / float(freq)
    t_end = n_cycles * P
    dt = P / float(coarse_pts_per_period)
    # small epsilon to include t_end
    t_coarse = np.arange(0, t_end + 1e-12, dt)

    # switching times
    switches = np.arange(n_cycles + 1) * P
    switches = np.clip(switches, 0.0, t_end)

    # halos around switches
    hw = halo_frac * P
    halos_list = []
    for tk in switches:
        a = max(0.0, tk - hw)
        b = min(t_end, tk + hw)
        if b > a:
            halos_list.append(np.linspace(a, b, halo_points))
    halos = np.concatenate(halos_list) if halos_list else np.array([], dtype=float)

    t_eval = np.unique(np.concatenate([t_coarse, switches, halos]))
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
cmax = 7.5e-9     # sites/cm²
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23     # 1/mol
partialPH2 = 1.0
beta = [0.5, 0.5]
V_app = -0.4
k = 10

k_V_RDS = 1e-10

# base kT and kH (for naming + magnitude)
k_T_base = k_V_RDS * 100
k_H_base = k_V_RDS * 100

freq_array = np.array(np.logspace(-9, 9, 41))

# dG values, static volcano
dGmin_eV = 0.0  # eV
dGmax_eV = 0.20   # eV

# dG values, dynamic volcano
dGmin_dynamic = 0.03  # eV
dGmax_dynamic = 0.17  # eV

# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

###############################################################################
# STORAGE FOR ALL MECHANISMS
###############################################################################

# Dynamic results per mechanism (for Excel)
dyn_results_all = {"VT": [], "VH": []}

# Dynamic overlay storage per mechanism (for volcano plots)
overlay_storage = {
    "VT": {"rT": {}, "rH": {}, "curr": {}},
    "VH": {"rT": {}, "rH": {}, "curr": {}},
}

# For static volcano summary
static_summary_rows = []  # will hold dicts with mechanism, GHad, avg_rT/r_H, current

# Remember last maxstep per mechanism for titles
last_maxstep_per_mech = {"VT": None, "VH": None}

###############################################################################
# === DYNAMIC GHad(t) SIMULATION ===
###############################################################################

if do_dynamic_ghad:

    print("\nRunning dynamic GHad(t) simulation...")
    # Time-varying GHad values (in J)
    dGmin = dGmin_dynamic * Avo * conversion_factor
    dGmax = dGmax_dynamic * Avo * conversion_factor

    for mechanism_choice in mechanisms_to_run:
        mech_label = MECH_LABEL[mechanism_choice]
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

        for freq in freq_array:
            print(f"\nRunning simulation with period = {freq:.2e} Hz...")

            # time spacing
            t, max_time = make_t_eval(freq, n_cycles=20,
                                      coarse_pts_per_period=200,
                                      halo_frac=0.10, max_eq_pts=500,
                                      halo_points=100)
            duration = [0, max_time]

            # keep the solver from skipping over switch neighborhoods
            P = 2.0 / freq
            maxstep = 1 / (freq * 1e2)
            print(f"Max Step: {maxstep:.2e}")

            def dGvt(t_local):
                return (dGmin) + (dGmax - dGmin) * (np.tanh(k * np.sin(2 * np.pi * freq * t_local)) + 1) / 2

            # static potential
            def potential(t_local):
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
            def rates_r0(t_local, theta):
                GHad_local = dGvt(t_local)
                theta = np.asarray(theta)
                thetaA_star, thetaA_H = theta
                V_local = potential(t_local)
                U_V, U_H = eqpot(theta, GHad_local)

                # Volmer Rate Equation
                r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
                    * np.exp(beta[0] * GHad_local / RT) * (
                        np.exp(-(beta[0]) * F * (V_local - U_V) / RT)
                        - np.exp((1 - beta[0]) * F * (V_local - U_V) / RT))

                r_T = 0.0
                if mechanism_choice == 0:  # VT
                    T_1 = (thetaA_H ** 2)
                    T_2 = (partialPH2 * (thetaA_star ** 2)
                           * np.exp((-2 * GHad_local) / RT))
                    r_T = k_T * (T_1 - T_2)

                r_H = 0.0
                if mechanism_choice == 1:  # VH
                    j1 = k_H * np.exp(-beta[1] * GHad_local / RT) * \
                        thetaA_star ** beta[1] * \
                        thetaA_H ** (1 - beta[1])
                    exp21 = np.exp(-beta[1] * F * (V_local - U_H) / RT)
                    exp22 = np.exp((1 - beta[1]) * F * (V_local - U_H) / RT)
                    r_H = j1 * (exp21 - exp22)

                return r_V, r_T, r_H

            def theta_H_eq_dynamic(GHad_init, V_local, mech_choice):
                # bracket for theta, solution should be between 1e-9 and 1 - 1e-9
                lo, hi = 1e-9, 1 - 1e-9

                def f(thetaH):
                    theta = np.array([1 - thetaH, thetaH])
                    rV_local, rT_local, rH_local = rates_r0(0, theta)
                    if mech_choice == 0:
                        return rV_local - 2 * rT_local
                    else:
                        return rV_local - rH_local

                sol = root_scalar(f, bracket=[lo, hi])
                return sol.root

            # Initial coverage of Hads, inside loop so that it starts fresh each time
            thetaA_H0_dynamic = theta_H_eq_dynamic(dGmin, V_app, mechanism_choice)
            thetaA_Star0_dynamic = 1.0 - thetaA_H0_dynamic  # Initial coverage of empty sites
            theta0_dynamic = [thetaA_Star0_dynamic, thetaA_H0_dynamic]

            def sitebal(t_local, theta):
                r_V_local, r_T_local, r_H_local = rates_r0(t_local, theta)
                if mechanism_choice == 0:
                    thetaStar_rate_VT = (-r_V_local + (2 * r_T_local)) / cmax
                    thetaH_rate_VT = (r_V_local - (2 * r_T_local)) / cmax
                    dthetadt_local = [thetaStar_rate_VT, thetaH_rate_VT]
                else:
                    theta_star_rate = r_H_local - r_V_local
                    theta_H_rate = r_V_local - r_H_local
                    dthetadt_local = [theta_star_rate / cmax, theta_H_rate / cmax]
                return dthetadt_local

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
                "period": 2 / (freq),
                "freq": float(freq),
                "t": t.copy(),
                "curr": curr_dynamic.copy(),
                "thetaH": thetaH_array.copy(),
                "GHad_eV": GHad_t_eV.copy(),
                "Average Current": avg_curr,
                "maxstep": maxstep,
            })

            # OPTIONAL time-domain plots (same as your original; keep or comment out)
            P_local = 2.0 / freq
            cycles_to_plot = 5
            t_start = 0
            t_end_plot = cycles_to_plot * P_local
            mask = (t >= t_start) & (t <= t_end_plot)

            average_rT = np.average(r_T_vals)
            average_rH = np.average(r_H_vals)

# =============================================================================
#             if mechanism_choice == 0:
#                 plt.figure(figsize=(8, 5))
#                 plt.plot(t[mask], r_T_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
#                 plt.axhline(y=average_rT, color="red", linestyle="--", linewidth=2,
#                             label=f"Average rT = {average_rT:.2e}")
#                 plt.xlabel("Time (s)")
#                 plt.ylabel(r"$r_T$ (mol/cm²·s)")
#                 plt.title(f"r_T vs Time at {freq:.2e} Hz, kV = {k_V}, maxstep = {maxstep:.2e}")
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.tight_layout()
#                 plt.show()
# 
#             if mechanism_choice == 1:
#                 plt.figure(figsize=(8, 5))
#                 plt.plot(t[mask], r_H_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
#                 plt.axhline(y=average_rH, color="red", linestyle="--", linewidth=2,
#                             label=f"Average rH = {average_rH:.2e}")
#                 plt.xlabel("Time (s)")
#                 plt.ylabel(r"$r_H$ (mol/cm²·s)")
#                 plt.title(f"r_H vs Time at {freq:.2e} Hz, kV = {k_V}, maxstep = {maxstep:.2e}")
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.tight_layout()
#                 plt.show()
# 
#             # Coverage vs time
#             plt.figure(figsize=(12, 10))
#             plt.plot(t[mask], thetaH_array[mask], label=f'Theta_H Coverage ({freq:.2e} Hz)')
#             plt.xlabel("Time (s)")
#             plt.ylabel(r"$\theta_H$")
#             plt.title(f'Coverage vs Time, {freq:.2e} Hz ({mech_label})')
#             plt.grid(True, alpha=0.4)
#             plt.legend()
#             plt.show()
# 
#             # rV vs time
#             plt.figure(figsize=(12, 10))
#             plt.plot(t[mask], r_V_vals[mask], label=f'rV ({freq:.2e} Hz)')
#             plt.xlabel("Time (s)")
#             plt.ylabel(r"$r_V$")
#             plt.title(f'rV vs Time, {freq:.2e} Hz ({mech_label})')
#             plt.grid(True, alpha=0.4)
#             plt.legend()
#             plt.show()
# =============================================================================

            # store last maxstep for this mechanism
            last_maxstep_per_mech[mech_label] = maxstep

        # save dynamic results & overlays for this mechanism
        dyn_results_all[mech_label] = dyn_results
        overlay_storage[mech_label]["rT"] = dynamic_overlay_by_freq
        overlay_storage[mech_label]["rH"] = dynamic_overlay_by_freq_rH
        overlay_storage[mech_label]["curr"] = dynamic_overlay_by_freq1

###############################################################################
# === STATIC VOLCANO PLOTS (VT & VH, with overlays)
###############################################################################

# We will store per-mechanism static volcano data
static_results_VT_rT = []   # (GHad_eV, avg_rT)
static_results_VT_curr = [] # (GHad_eV, avg_curr)
static_results_VH_rH = []   # (GHad_eV, avg_rH)
static_results_VH_curr = [] # (GHad_eV, avg_curr)

if do_static_volcano:

    print("\nRunning static volcano plot...\n")

    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 16)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor

    t_end = 5
    t_eval = np.linspace(0, t_end, 200)

    for mechanism_choice in mechanisms_to_run:
        mech_label = MECH_LABEL[mechanism_choice]
        print(f"=== STATIC VOLCANO FOR {mech_label} ===")

        # set k's like in dynamic
        if mechanism_choice == 0:   # VT
            k_V = k_V_RDS
            k_T = k_T_base
            k_H = 0.0
        else:                       # VH
            k_V = k_V_RDS
            k_T = 0.0
            k_H = k_H_base

        # Shared static rate expression (depends on mechanism)
        def rates_r0_static(theta, GHad_local, mechanism):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta

            U_V = (-GHad_local / F) + (RT * np.log(thetaA_star / thetaA_H)) / F

            # Volmer
            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
                * np.exp(beta[0] * GHad_local / RT) * (
                    np.exp(-(beta[0]) * F * (V_app - U_V) / RT)
                    - np.exp((1 - beta[0]) * F * (V_app - U_V) / RT))

            if mechanism == "VT":
                r_T = k_T * ((thetaA_H ** 2) -
                             partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad_local) / RT))
                r_H = 0.0
            else:  # VH
                U_H = (GHad_local / F) + (RT / F) * np.log(thetaA_H / thetaA_star)
                j1 = k_H * np.exp(-beta[1] * GHad_local / RT) \
                    * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                r_H = j1 * (
                    np.exp(-beta[1] * F * (V_app - U_H) / RT)
                    - np.exp((1 - beta[1]) * F * (V_app - U_H) / RT)
                )
                r_T = 0.0

            return r_V, r_T, r_H

        # equilibrium theta for static
        def theta_H_eq_static(GHad_local, mechanism):
            lo, hi = 1e-9, 1 - 1e-9

            def f(thetaH):
                theta = np.array([1 - thetaH, thetaH])
                rV, rT, rH = rates_r0_static(theta, GHad_local, mechanism)
                if mechanism == "VT":
                    return rV - 2 * rT
                else:
                    return rV - rH

            sol = root_scalar(f, bracket=[lo, hi])
            return sol.root

        for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):

            if mech_label == "VT":
                mechanism_str = "VT"
            else:
                mechanism_str = "VH"

            thetaA_H0_static = theta_H_eq_static(GHad, mechanism_str)
            thetaA_Star0_static = 1.0 - thetaA_H0_static
            theta0_static = [thetaA_Star0_static, thetaA_H0_static]

            def sitebal_static(t_static, theta):
                r_V, r_T, r_H = rates_r0_static(theta, GHad, mechanism_str)
                if mechanism_str == "VT":
                    thetaStar_rate = (-r_V + (2 * r_T)) / cmax
                    thetaH_rate = (r_V - (2 * r_T)) / cmax
                else:
                    thetaStar_rate = (r_H - r_V) / cmax
                    thetaH_rate = (r_V - r_H) / cmax
                return [thetaStar_rate, thetaH_rate]

            soln = solve_ivp(sitebal_static, (0, t_end), theta0_static,
                             t_eval=t_eval, method='BDF')

            r0_vals = np.array([rates_r0_static(theta, GHad, mechanism_str)
                                for theta in soln.y.T])
            rV_vals = r0_vals[:, 0]
            rT_vals = r0_vals[:, 1]
            rH_vals = r0_vals[:, 2]

            curr_static = rV_vals * -F * 1000  # mA/cm²
            average_current = np.abs(np.average(curr_static))
            averagerT = np.average(rT_vals[5:])
            average_rH = np.average(rH_vals[5:])

            thetaA_H = soln.y[1, :]
            avg_thetaH = np.average(thetaA_H[5:])
            print(f'Mechanism {mechanism_str}, GHad = {GHad_eV:.3f} eV → Average Coverage: {avg_thetaH:.3f}')

            if mechanism_str == "VT":
                print(f"Average rT for {GHad_eV}:", averagerT)
                static_results_VT_rT.append((GHad_eV, averagerT))
                static_results_VT_curr.append((GHad_eV, average_current))
                static_summary_rows.append({
                    "Mechanism": "VT",
                    "GHad (eV)": GHad_eV,
                    "Average r_T (mol/cm²·s)": averagerT,
                    "Average r_H (mol/cm²·s)": 0.0,
                    "Average Current (mA/cm²)": average_current
                })
            else:
                print(f"Average rH for {GHad_eV}:", average_rH)
                static_results_VH_rH.append((GHad_eV, average_rH))
                static_results_VH_curr.append((GHad_eV, average_current))
                static_summary_rows.append({
                    "Mechanism": "VH",
                    "GHad (eV)": GHad_eV,
                    "Average r_T (mol/cm²·s)": 0.0,
                    "Average r_H (mol/cm²·s)": average_rH,
                    "Average Current (mA/cm²)": average_current
                })
    
    colors = plt.cm.turbo(np.linspace(0, 1, len(freq_array)))
    freq_colors = dict(zip(freq_array, colors))
    
# =============================================================================
#     # =======================
#     # ==== PLOTTING VT ======
#     # =======================
#     if static_results_VT_rT:
#         GHad_vals, avg_rT_vals = zip(*static_results_VT_rT)
#         # use VT overlays + last maxstep
#         dynamic_overlay_by_freq = overlay_storage["VT"]["rT"]
#         dynamic_overlay_by_freq1 = overlay_storage["VT"]["curr"]
#         maxstep_VT = last_maxstep_per_mech["VT"] if last_maxstep_per_mech["VT"] is not None else 0.0
# 
#         # Volcano plot (rT) - EXACT STYLE FROM ORIGINAL
#         plt.figure(figsize=(10, 8))
#         plt.plot(GHad_vals, avg_rT_vals, label='Static GHad Scan', marker='o', color="blue")
# 
#         if dynamic_overlay_by_freq:
#             for f in freq_array:
#                 if f not in dynamic_overlay_by_freq:
#                     continue
#                 pts = dynamic_overlay_by_freq[f]
#                 xs = [p[0] for p in pts]
#                 ys = [p[1] for p in pts]
# 
#                 # scatter markers
#                 plt.scatter(xs, ys,
#                             s=120,
#                             facecolors="none",
#                             edgecolors=freq_colors.get(f, None),
#                             linewidths=2,
#                             zorder=3)
# 
#                 # inline text labels
#                 for x, y in zip(xs, ys):
#                     plt.text(
#                         x, y,
#                         f"{f:.2e} Hz",  # label like original
#                         fontsize=16, weight="bold",
#                         color=freq_colors.get(f, "black"),
#                         ha="left", va="bottom",
#                         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
#                     )
# 
#         plt.xlabel("GHad (eV)")
#         plt.ylabel("Average $\\mathbf{r_T}$ (mol/cm²·s)")
#         plt.title(f"$\\mathbf{{r_T}}$ vs GHad, V = {V_app} V, maxstep = {maxstep_VT:.2e}")
#         ax = plt.gca()
#         ax.xaxis.set_major_locator(MultipleLocator(0.05))
#         plt.tight_layout()
#         plt.show()
# 
#         # Volcano plot (Current) for VT
#         if static_results_VT_curr:
#             GHad_vals_curr, avg_curr_vals_VT = zip(*static_results_VT_curr)
#             plt.figure(figsize=(10, 8))
#             plt.plot(GHad_vals_curr, avg_curr_vals_VT,
#                      label='Static GHad Scan', marker='o', color="blue")
# 
#             if dynamic_overlay_by_freq1:
#                 for f in freq_array:
#                     if f not in dynamic_overlay_by_freq1:
#                         continue
#                     pts = dynamic_overlay_by_freq1[f]
#                     xs = [p[0] for p in pts]  # GHad (eV)
#                     ys = [p[1] for p in pts]  # Current
# 
#                     plt.scatter(xs, ys,
#                                 s=120,
#                                 facecolors="none",
#                                 edgecolors=freq_colors.get(f, None),
#                                 linewidths=2,
#                                 zorder=3)
# 
#                     for x, y in zip(xs, ys):
#                         plt.text(
#                             x, y,
#                             f"{f:.2e} Hz",
#                             fontsize=16, weight="bold",
#                             color=freq_colors.get(f, "black"),
#                             ha="left", va="bottom",
#                             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
#                         )
# 
#             plt.xlabel("GHad (eV)")
#             plt.ylabel("Average |Current Density| (mA/cm²)")
#             plt.title(f"Volcano Plot: Average Current vs GHad, V = {V_app} V (VT)")
#             ax = plt.gca()
#             ax.xaxis.set_major_locator(MultipleLocator(0.05))
#             plt.tight_layout()
#             plt.show()
# 
#     # =======================
#     # ==== PLOTTING VH ======
#     # =======================
#     if static_results_VH_rH:
#         GHad_vals, avg_rH_vals = zip(*static_results_VH_rH)
#         dynamic_overlay_by_freq_rH = overlay_storage["VH"]["rH"]
#         dynamic_overlay_by_freq1_VH = overlay_storage["VH"]["curr"]
#         maxstep_VH = last_maxstep_per_mech["VH"] if last_maxstep_per_mech["VH"] is not None else 0.0
# 
#         # ============================
#         # VH rH Volcano Plot
#         # ============================
#         plt.figure(figsize=(10, 8))
#         plt.plot(GHad_vals, avg_rH_vals, label='Static GHad Scan', marker='o', color="blue")
# 
#         if dynamic_overlay_by_freq_rH:
#             for f in freq_array:
#                 if f not in dynamic_overlay_by_freq_rH:
#                     continue
# 
#                 pts = dynamic_overlay_by_freq_rH[f]
#                 xs = [p[0] for p in pts]
#                 ys = [p[1] for p in pts]
# 
#                 plt.scatter(xs, ys,
#                             s=120,
#                             facecolors="none",
#                             edgecolors=freq_colors.get(f, None),
#                             linewidths=2,
#                             zorder=3)
#                 for x, y in zip(xs, ys):
#                     plt.text(
#                         x, y,
#                         f"{f:.2e} Hz",
#                         fontsize=16, weight="bold",
#                         color=freq_colors.get(f, "black"),
#                         ha="left", va="bottom",
#                         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
#                     )
# 
#         plt.xlabel("GHad (eV)")
#         plt.ylabel("Average $\\mathbf{r_H}$ (mol/cm²·s)")
#         plt.title(f"$\\mathbf{{r_H}}$ vs GHad, V = {V_app} V, maxstep = {maxstep_VH:.2e}")
#         ax = plt.gca()
#         ax.xaxis.set_major_locator(MultipleLocator(0.05))
#         plt.tight_layout()
#         plt.show()
# 
#         # ============================
#         # VH Current Volcano Plot
#         # ============================
#         if static_results_VH_curr:
#             GHad_vals_curr, avg_curr_vals_VH = zip(*static_results_VH_curr)
# 
#             plt.figure(figsize=(10, 8))
#             plt.plot(GHad_vals_curr, avg_curr_vals_VH,
#                      label='Static GHad Scan', marker='o', color="blue")
# 
#             if dynamic_overlay_by_freq1_VH:
#                 for f in freq_array:
#                     if f not in dynamic_overlay_by_freq1_VH:
#                         continue
# 
#                     pts = dynamic_overlay_by_freq1_VH[f]
#                     xs = [p[0] for p in pts]
#                     ys = [p[1] for p in pts]
# 
#                     plt.scatter(xs, ys,
#                                 s=120,
#                                 facecolors="none",
#                                 edgecolors=freq_colors.get(f, None),
#                                 linewidths=2,
#                                 zorder=3)
# 
#                     for x, y in zip(xs, ys):
#                         plt.text(
#                             x, y,
#                             f"{f:.2e} Hz",
#                             fontsize=16, weight="bold",
#                             color=freq_colors.get(f, "black"),
#                             ha="left", va="bottom",
#                             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1)
#                         )
# 
#             plt.xlabel("GHad (eV)")
#             plt.ylabel("Average |Current Density| (mA/cm²)")
#             plt.title(f"Volcano Plot: Average Current vs GHad, V = {V_app} V (VH)")
#             ax = plt.gca()
#             ax.xaxis.set_major_locator(MultipleLocator(0.05))
#             plt.tight_layout()
#             plt.show()
# =============================================================================
######################### VT Freq Sweep Plot ################################

mech_T = "VT"
quantity_T = "rT"   # choose: "rT", "rH", or "curr"

freqs_T = []
avg_vals_T = []

for f, pts in overlay_storage[mech_T][quantity_T].items():
    val_weak = pts[0][1]    # weak binding value
    val_strong = pts[1][1]  # strong binding value
    avg_val = 0.5 * (val_weak + val_strong)

    freqs_T.append(f)
    avg_vals_T.append(avg_val)

# Sort by frequency (important for log plots)
freqs_T = np.array(freqs_T)
avg_vals_T = np.array(avg_vals_T)
order = np.argsort(freqs_T)

freqs_T = freqs_T[order]
avg_vals_T = avg_vals_T[order]

# Plot
plt.figure(figsize=(9, 7))
plt.plot(freqs_T, avg_vals_T, marker='o', linewidth=2)
ymin_T = avg_vals_T.min()
ymax_T = avg_vals_T.max()

pad_T = 0.05 * (ymax_T - ymin_T)   # 5% padding

plt.ylim(ymin_T - pad_T, ymax_T + pad_T)
plt.xscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Average rT (mol/cm2*s)")
plt.title(f"Average rT vs Frequency, at V={V_app}, dGmin={dGmin_dynamic}, dGmax={dGmax_dynamic}")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

mech_H = "VH"
quantity_H = "rH"   # choose: "rT", "rH", or "curr"

freqs_H = []
avg_vals_H = []

for f, pts in overlay_storage[mech_H][quantity_H].items():
    val_weak = pts[0][1]    # weak binding value
    val_strong = pts[1][1]  # strong binding value
    avg_val = 0.5 * (val_weak + val_strong)

    freqs_H.append(f)
    avg_vals_H.append(avg_val)

############################## VH FREQ SWEEP PLOT ######################################

# Sort by frequency (important for log plots)
freqs_H = np.array(freqs_H)
avg_vals_H = np.array(avg_vals_H)
order_H = np.argsort(freqs_H)

freqs_H = freqs_H[order]
avg_val_Hs = avg_vals_H[order]

# Plot
plt.figure(figsize=(9, 7))
plt.plot(freqs_H, avg_vals_H, marker='o', linewidth=2)
ymin_H = avg_vals_H.min()
ymax_H = avg_vals_H.max()

pad_H = 0.05 * (ymax_H - ymin_H)   # 5% padding

plt.ylim(ymin_H - pad_H, ymax_H + pad_H)
plt.xscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Average rH (mol/cm2*s)")
plt.title(f"Average rH vs Frequency, at V={V_app}, dGmin={dGmin_dynamic}, dGmax={dGmax_dynamic}")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

################################## COMBINED FREQ SWEEP PLOT ####################################
#combined
plt.figure(figsize=(9, 7))
plt.plot(freqs_H, avg_vals_H, marker='o', linewidth=2, label='Average rH Values')
plt.plot(freqs_T, avg_vals_T, marker='o', linewidth=2, label='Average rT Values')
#plt.ylim((min(ymin_T, ymin_H) - max(pad_H, pad_T)), (max(ymax_T, ymax_H) + max(pad_H, pad_T)))
plt.xscale('log')
plt.ylabel('Average Rate (mol/cm2*s)')
plt.legend()
plt.grid()
plt.title(f'Average Rate vs Frequency, at V={V_app}, dGmin={dGmin_dynamic}, dGmax={dGmax_dynamic}')
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10))
# =============================================================================
# ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[]))
# ax.xaxis.set_major_formatter(LogFormatterSciNotation())
# ax.set_xlim(freq_array.min(), freq_array.max())
# =============================================================================
plt.show()