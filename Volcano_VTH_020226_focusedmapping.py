import numpy as np
import os
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import time

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

def make_output_filename(kV, kT, kH, freq_phys=None, beta=None,
                         dGmin=None, dGmax=None, voltage=None,
                         t_switching=None, cycles_to_run=None,
                         base="dynamic_simulation_output.xlsx"):
    """
    Build a unique filename string with parameters included.
    kV, kT, kH always appear in the filename, regardless of mechanism.
    """
    
    k_str = f"kV={kV:.2e}_kT={kT:.2e}_kH={kH:.2e}"

    filename = (
        f"sim_k_{k_str}_freq_phys_cycles_{cycles_to_run}"
        f"_dG_{dGmin:.2f}-{dGmax:.2f}eV__.xlsx"
    )

    final_filename = filename
    counter = 1
    while os.path.exists(final_filename):
        final_filename = filename.replace(".xlsx", f"_{counter}.xlsx")
        counter += 1
    return final_filename

########################################  Time Function ###################################################

def make_t_eval(freq_phys, n_cycles=20, pts_per_freq_phys = 100):
    P = 1.0 / float(freq_phys)
    t_end = n_cycles * P
    dt = P / pts_per_freq_phys
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
V_app = np.array([-0.1, -0.2, -0.3])
k = 95
a_tol = 1e-14
r_tol = 1e-8
phi = -np.pi/2
duty = 0.8   #0.5 is 50/50 split, 0.1 -> majority dGmin | 0.9 -> majority dGmax
cycles_to_run = 1e2

k_V_RDS = 1e-10

# base kT and kH (for naming + magnitude)
k_T_base = k_V_RDS * 20000
k_H_base = k_V_RDS * 500

freq_phys = 1e1
freq_phys_norm = freq_phys / (k_V_RDS / cmax)

# dG values, static volcano
dGmin_eV = -0.1  # eV
dGmax_eV = 0.2   # eV

# dG values, dynamic
dGmin_dynamic = 0.05  # eV
dGmax_dynamic = 0.15  # eV

# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

###############################################################################
# STORAGE FOR ALL MECHANISMS
###############################################################################

avg_rV_freq_phys = []
avg_rT_freq_phys = []
avg_rH_freq_phys = []

# Dynamic results per mechanism (for Excel)
dyn_results_all = {"VT": [], "VH": []}

# Dynamic overlay storage per mechanism (for volcano plots)
overlay_storage = {
    "VT": {"rT": {}, "rH": {}, "rV": {}, "avg": {}},
    "VH": {"rT": {}, "rH": {}, "rV": {}, "avg": {}},
}

voltage_trace = {}

# For static volcano summary
static_summary_rows = []  # will hold dicts with mechanism, GHad, avg_rT/r_H, current

# Remember last maxstep per mechanism for titles
last_maxstep_per_mech = {"VT": None, "VH": None}

###############################################################################
# === DYNAMIC GHad(t) SIMULATION ===
###############################################################################

start_time = time.perf_counter()
if do_dynamic_ghad:

    print("\nRunning dynamic GHad(t) simulation...")

    # =========================
    # dGmin/dGmax sweep settings
    # =========================
    dG_lo, dG_hi, dG_step = -0.10, 0.40, 0.01
    dG_vals = np.round(np.arange(dG_lo, dG_hi + 1e-12, dG_step), 2)
    n_dG = len(dG_vals)
    
    # heatmaps[mech_label][freq_phys] = Z matrix
    heatmaps = {"VT": {}, "VH": {}}
    
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
        dynamic_overlay_by_freq_phys = {}    # r_T overlay
        dynamic_overlay_by_freq_phys1 = {}   # current overlay
        dynamic_overlay_by_freq_phys_rH = {} # r_H overlay

        avg_currents = []

        # Lists per freq_physuency
        avg_currents_dGmin = []
        avg_currents_dGmax = []
        avg_rV_dGmin = []
        avg_rV_dGmax = []
        avg_rT_dGmin = []
        avg_rT_dGmax = []
        avg_rH_dGmin = []
        avg_rH_dGmax = []

        dyn_results = []
        
        
        vh_trace_store = {}


        for V in V_app:
            print(f"\nRunning simulation with voltage = {V}..")

            # time spacing
            t, max_time = make_t_eval(freq_phys, n_cycles=cycles_to_run, pts_per_freq_phys=10)            
            duration = [0, max_time]

            # keep the solver from skipping over switch neighborhoods
            P = 1.0 / freq_phys
            maxstep = 1 / (freq_phys * 1e3)
            print(f"Max Step: {maxstep}")

            def dGvt(t):
                return (dGmin) + (dGmax - dGmin) * (np.tanh(k * (np.sin(2*np.pi * freq_phys * t - phi) - np.sin(np.pi * (0.5 - duty)))) + 1) / 2

            # static potential
            def potential(t):
                return V

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
            thetaA_H0_dynamic = theta_H_eq_dynamic(dGvt(0), V, mechanism_choice)
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

            GHad_range = dGmax - dGmin
            mask_min = (np.abs(GHad_t_J - dGmin) < 0.2 * GHad_range)
            mask_max = (np.abs(GHad_t_J - dGmax) < 0.2 * GHad_range)
            
            voltage_trace[f"{V:.2f}"]  = {
                "r_V": r_V_vals.copy(),
                "r_T": r_T_vals.copy(),
                "r_H": r_H_vals.copy(),
                "t": t.copy(),
                "thetaH": thetaH_array.copy(),
                "GHad_eV": GHad_t_eV.copy(),
                "current": curr_dynamic.copy(),
            }

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

            # Absolute value of rV at GHad min/max
            avg_rV_at_dGmin = np.average(r_V_vals[mask_min])
            avg_rV_at_dGmax = np.average(r_V_vals[mask_max])
            avg_rV_dGmin.append(avg_rV_at_dGmin)
            avg_rV_dGmax.append(avg_rV_at_dGmax)

            # Save them for overlay plotting (per-freq_physuency)
            dynamic_overlay_points.append((dGmin_dynamic, avg_rV_at_dGmin))
            dynamic_overlay_points.append((dGmax_dynamic, avg_rV_at_dGmax))

            dynamic_overlay_by_freq_phys[freq_phys] = [
                (dGmin_dynamic, float(average_rT_dGmin)),
                (dGmax_dynamic, float(average_rT_dGmax)),
            ]
            dynamic_overlay_by_freq_phys1[freq_phys] = [
                (dGmin_dynamic, float(avg_rV_at_dGmin)),
                (dGmax_dynamic, float(avg_rV_at_dGmax)),
            ]
            dynamic_overlay_by_freq_phys_rH[freq_phys] = [
                (dGmin_dynamic, float(average_rH_dGmin)),
                (dGmax_dynamic, float(average_rH_dGmax)),
            ]

            dyn_results.append({
                "r_T": r_T_vals,
                "r_H": r_H_vals,
                "rV": r_V_vals,
                "period": 1 / (freq_phys),
                "freq_phys": float(freq_phys),
                "t": t.copy(),
                "curr": curr_dynamic.copy(),
                "thetaH": thetaH_array.copy(),
                "GHad_eV": GHad_t_eV.copy(),
                "Average Current": avg_curr,
                "maxstep": maxstep,
            })

            
            #######################################################################
            # HOW TO USE IT IN *YOUR* SCRIPT
            #######################################################################
            # 1) Add a dict to store per-freq_physuency traces for VH (inside dynamic section)
            #    e.g. before the for freq_phys in freq_phys loop:
            #
            # vh_trace_store = {}
            #
            # 2) After you compute r_V_vals, theta_at_t, GHad_t_J, GHad_t_eV in the VH loop,
            #    store them:
            #
            # if mech_label == "VH":
            #     vh_trace_store[freq_phys] = {
            #         "t": t.copy(),
            #         "theta_at_t": theta_at_t.copy(),   # (2,N)
            #         "GHad_J": GHad_t_J.copy(),
            #         "GHad_eV": GHad_t_eV.copy(),
            #         "rV": r_V_vals.copy(),
            #         "RT": RT,
            #         "F": F,
            #     }
            #
            # 3) After the dynamic simulation finishes (after loops), call:
            #
            # plot_regime_diagnostics(vh_trace_store, list(vh_trace_store.keys()),
            #                         title_prefix="VH: G(t) vs θH(t) → ηV(t) → rV(t)",
            #                         n_cycles_plot=5, V_app=V_app)
            #
            #######################################################################
            
            
            t_averaging_start = 0.4
            t_averaging_end = cycles_to_run * P
            mask_time = (t >= t_averaging_start) & (t <= t_averaging_end)
            thetaH_dynamic = thetaH_array[mask_time]
            thetaH_dynamic_averaging = np.average(thetaH_dynamic)
            
            
            # OPTIONAL time-domain plots (same as your original; keep or comment out)
            P = 1.0 / freq_phys
            cycles_to_plot_binding = cycles_to_run
            cycles_to_plot_coverage = 20
            t_start = 0
            t_end_plot = cycles_to_plot_coverage * P
            t_end_binding = cycles_to_plot_binding * P
            mask = (t >= t_start) & (t <= t_end_plot)
            mask_binding = (t >= t_start) & (t <= t_end_binding)

            average_rT_dynamic = np.average(r_T_vals)
            average_rH_dynamic = np.average(r_H_vals)
            average_rV_dynamic = np.average(r_V_vals)
            
# =============================================================================
#             # Binding Energy vs time
#             plt.figure(figsize=(12, 10))
#             plt.plot(t[mask_binding], GHad_t_eV[mask_binding], label=f'Theta_H Coverage ({freq_phys:.2e} Hz)')
#             plt.xlabel("Time (s)")
#             plt.ylabel(r"Binding Energy, eV")
#             plt.title(f'Binding Energy vs Time, {freq_phys:.2e} Hz ({mech_label}), phi={phi:.2f}')
#             plt.grid(True, alpha=0.4)
#             plt.legend()
#             plt.show()
# =============================================================================
            
            # Coverage vs time
            plt.figure(figsize=(12, 10))
            plt.plot(t[mask], thetaH_array[mask], label=f'Theta_H Coverage ({freq_phys:.2e} Hz)')
            plt.axhline(y=thetaH_dynamic_averaging, color="red", linestyle="--", linewidth=2, label=f"Max Static rT = {thetaH_dynamic_averaging:.2e}")
            plt.xlabel("Time (s)")
            plt.ylabel(r"$\theta_H$")
            plt.title(f'Coverage vs Time, {freq_phys:.2e} Hz ({mech_label}), phi={phi:.2f}')
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.show()

# ================================================ =============================
#             if mechanism_choice == 0:
#                 plt.figure(figsize=(8, 5))
#                 plt.plot(t[mask], r_T_vals[mask], label=f"{freq_phys:.2e} Hz", linewidth=1.8)
#                 plt.axhline(y=average_rT, color="red", linestyle="--", linewidth=2,
#                             label=f"Average rT = {average_rT:.2e}")
#                 plt.xlabel("Time (s)")
#                 plt.ylabel(r"$r_T$ (mol/cm²·s)")
#                 plt.title(f"r_T vs Time at {freq_phys:.2e} Hz, kV = {k_V}, maxstep = {maxstep:.2e}")
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.tight_layout()
#                 plt.show()
# 
#             if mechanism_choice == 1:
#                 plt.figure(figsize=(8, 5))
#                 plt.plot(t[mask], r_H_vals[mask], label=f"{freq_phys:.2e} Hz", linewidth=1.8)
#                 plt.axhline(y=average_rH, color="red", linestyle="--", linewidth=2,
#                             label=f"Average rH = {average_rH:.2e}")
#                 plt.xlabel("Time (s)")
#                 plt.ylabel(r"$r_H$ (mol/cm²·s)")
#                 plt.title(f"r_H vs Time at {freq_phys:.2e} Hz, kV = {k_V}, maxstep = {maxstep:.2e}")
#                 plt.legend()
#                 plt.grid(True, alpha=0.3)
#                 plt.tight_layout()
#                 plt.show()
# 
#             # Coverage vs time
#             plt.figure(figsize=(12, 10))
#             plt.plot(t[mask], thetaH_array[mask], label=f'Theta_H Coverage ({freq_phys:.2e} Hz)')
#             plt.xlabel("Time (s)")
#             plt.ylabel(r"$\theta_H$")
#             plt.title(f'Coverage vs Time, {freq_phys:.2e} Hz ({mech_label})')
#             plt.grid(True, alpha=0.4)
#             plt.legend()
#             plt.show()
# 
#             # rV vs time
#             plt.figure(figsize=(12, 10))
#             plt.plot(t[mask], r_V_vals[mask], label=f'rV ({freq_phys:.2e} Hz)')
#             plt.xlabel("Time (s)")
#             plt.ylabel(r"$r_V$")
#             plt.title(f'rV vs Time, {freq_phys:.2e} Hz ({mech_label})')
#             plt.grid(True, alpha=0.4)
#             plt.legend()
#             plt.show()
# =============================================================================

            # store last maxstep for this mechanism
            last_maxstep_per_mech[mech_label] = maxstep
        
            overlay_storage[mech_label]["avg"][freq_phys] = {
                "average_rV": average_rV_dynamic,
                "average_rT": average_rT_dynamic,
                "average_rH": average_rH_dynamic,
            }

        # save dynamic results & overlays for this mechanism
        dyn_results_all[mech_label] = dyn_results
        overlay_storage[mech_label]["rT"] = dynamic_overlay_by_freq_phys
        overlay_storage[mech_label]["rH"] = dynamic_overlay_by_freq_phys_rH
        overlay_storage[mech_label]["rV"] = dynamic_overlay_by_freq_phys1


# ======================== EXCEL EXPORT (ALL TOGETHER) ========================
output_filename = os.path.join(
    make_output_filename(kV=k_V, kT=k_T, kH=k_H,
                         freq_phys=freq_phys, beta=beta,
                         dGmin=dGmin_dynamic, dGmax=dGmax_dynamic,
                         voltage=V_app, cycles_to_run=cycles_to_run)
)

with pd.ExcelWriter("voltage_traces.xlsx", engine="openpyxl") as writer:
    for V_str, traces in voltage_trace.items():
        V=float(V_str)
        df = pd.DataFrame({
            "Time (s)": traces["t"],
            "r_V (mol/cm²·s)": traces["r_V"],
            "r_T (mol/cm²·s)": traces["r_T"],
            "r_H (mol/cm²·s)": traces["r_H"],
            "θ_H": traces["thetaH"],
            "GHad (eV)": traces["GHad_eV"],
            "Current (mA/cm²)": traces["current"],
        })
        df.to_excel(writer, sheet_name=f"V={V} V", index=False)

print(f"\nAll results exported to Excel: {output_filename}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"The code block executed in {elapsed_time:.4f} seconds")

##### plotting coverage as a function of voltage
P = 1/freq_phys
cycles_to_plot = 20 * P
mask = (0.5 < t) & ( t <= cycles_to_plot)

fig, ax = plt.subplots(figsize=(12, 8))
for V, traces in voltage_trace.items():
    t = traces["t"]
    thetaH = traces["thetaH"]
    ax.plot(
        t[mask],
        thetaH[mask],
        label=f"Voltage = {V} V",
        linewidth=2
        )
ax.set_xlabel("Time (s)")
ax.set_ylabel("Theta H Coverage")
ax.set_title("Coverage vs Time Per Voltage")
ax.grid(True, alpha=0.4)
ax.legend()

plt.tight_layout()
plt.show() 
