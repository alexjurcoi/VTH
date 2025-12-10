import numpy as np
import os
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "legend.fontsize": 16,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

###############################################################################
# MECHANISM MODE SELECTION (VT, VH, BOTH)
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

if mech_mode == 0:
    mechanisms_to_run = ["VT"]       # VT only
elif mech_mode == 1:
    mechanisms_to_run = ["VH"]       # VH only
else:
    mechanisms_to_run = ["VT", "VH"] # BOTH

MECH_TO_INT = {"VT": 0, "VH": 1}

###############################################################################
# FILE NAMING FUNCTION (ALWAYS INCLUDES kV, kT, kH)
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

###############################################################################
# TIME GRID FUNCTION
###############################################################################

def make_t_eval(freq, n_cycles=20,
                coarse_pts_per_period=4,
                halo_frac=0.05, max_eq_pts=500,
                halo_points=40):
    P = 2.0 / float(freq)
    t_end = n_cycles * P
    dt = P / float(coarse_pts_per_period)

    t_coarse = np.arange(0, t_end + 1e-12, dt)

    switches = np.arange(n_cycles + 1) * P
    switches = np.clip(switches, 0.0, t_end)

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

    if t_eval.size == 0 or t_eval[0] > 0.0:
        t_eval = np.insert(t_eval, 0, 0.0)
    if t_eval[-1] < t_end:
        t_eval = np.append(t_eval, t_end)

    return t_eval, t_end

###############################################################################
# GLOBAL PARAMETERS
###############################################################################

RT = 8.314 * 298  # J/mol
F = 96485.0       # C/mol
cmax = 7.5e-9     # sites/cm²
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23     # 1/mol
partialPH2 = 1.0
beta = [0.5, 0.5]
V_app = -0.1
k = 10

k_V_RDS = 1e-10
k_T_base = k_V_RDS * 1000  # VT Tafel
k_H_base = k_V_RDS * 1000  # VH Heyrovsky

freq_array = [k_V_RDS / 1e-6, k_V_RDS / 1e-10, k_V_RDS / 1e-11, k_V_RDS / 1e-12]

# dG values, static volcano
dGmin_eV = -0.10
dGmax_eV = 0.30

# dG values, dynamic volcano
dGmin_dynamic = 0.05
dGmax_dynamic = 0.15

###############################################################################
# USER PROMPTS: STATIC / DYNAMIC
###############################################################################

print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

###############################################################################
# STORAGE FOR ALL MECHANISMS
###############################################################################

# Dynamic results (for Excel / plots)
dyn_results_all = {"VT": [], "VH": []}

# Static summary table (for Excel)
static_summary_all = []

# Dynamic overlay points for volcano plots
dynamic_overlay_all = {
    "VT": {"rT": {}, "curr": {}, "rH": {}},
    "VH": {"rT": {}, "curr": {}, "rH": {}},
}

###############################################################################
# DYNAMIC GHad(t) SIMULATION
###############################################################################

if do_dynamic_ghad:

    print("\nRunning dynamic GHad(t) simulation...")

    dGmin = dGmin_dynamic * Avo * conversion_factor
    dGmax = dGmax_dynamic * Avo * conversion_factor

    for mech_label in mechanisms_to_run:
        mechanism_choice = MECH_TO_INT[mech_label]
        print(f"\n=== DYNAMIC SIMULATION FOR {mech_label} ===")

        # Set rate constants
        if mech_label == "VT":
            k_V = k_V_RDS
            k_T = k_T_base
            k_H = 0.0
        else:  # VH
            k_V = k_V_RDS
            k_T = 0.0
            k_H = k_H_base

        overlay = dynamic_overlay_all[mech_label]
        dyn_results = []

        for freq in freq_array:
            print(f"\nRunning simulation with frequency = {freq:.2e} Hz...")

            t, max_time = make_t_eval(freq, n_cycles=20,
                                      coarse_pts_per_period=200,
                                      halo_frac=0.10, max_eq_pts=500,
                                      halo_points=100)
            duration = [0, max_time]

            P = 2.0 / freq
            maxstep = 1 / (freq * 1e2)
            print(f"Max Step: {maxstep:.2e}")

            def dGvt(t_local):
                return (dGmin) + (dGmax - dGmin) * (np.tanh(k * np.sin(2 * np.pi * freq * t_local)) + 1) / 2

            def potential(t_local):
                return V_app

            def eqpot(theta, GHad):
                theta = np.asarray(theta)
                thetaA_star, thetaA_H = theta
                U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
                U_H = 0.0
                if mech_label == "VH":
                    U_H = (GHad / F) + (RT * np.log(thetaA_H / thetaA_star) / F)
                return U_V, U_H

            def rates_r0(t_local, theta):
                GHad = dGvt(t_local)
                theta = np.asarray(theta)
                thetaA_star, thetaA_H = theta
                V_local = potential(t_local)

                U_V, U_H = eqpot(theta, GHad)

                # Volmer step
                r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
                    * np.exp(beta[0] * GHad / RT) * (
                        np.exp(-(beta[0]) * F * (V_local - U_V) / RT)
                        - np.exp((1 - beta[0]) * F * (V_local - U_V) / RT))

                r_T = 0.0
                if mech_label == "VT":
                    T_1 = (thetaA_H ** 2)
                    T_2 = (partialPH2 * (thetaA_star ** 2)
                           * np.exp((-2 * GHad) / RT))
                    r_T = k_T * (T_1 - T_2)

                r_H = 0.0
                if mech_label == "VH":
                    j1 = k_H * np.exp(-beta[1] * GHad / RT) \
                        * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                    exp21 = np.exp(-beta[1] * F * (V_local - U_H) / RT)
                    exp22 = np.exp((1 - beta[1]) * F * (V_local - U_H) / RT)
                    r_H = j1 * (exp21 - exp22)

                return r_V, r_T, r_H

            def theta_H_eq_dynamic(GHad_init, V_local, mech_choice_label):
                lo, hi = 1e-9, 1 - 1e-9

                def f(thetaH):
                    theta = np.array([1 - thetaH, thetaH])
                    rV_local, rT_local, rH_local = rates_r0(0, theta)
                    if mech_choice_label == "VT":
                        return rV_local - 2 * rT_local
                    else:
                        return rV_local - rH_local

                sol = root_scalar(f, bracket=[lo, hi])
                return sol.root

            thetaA_H0_dynamic = theta_H_eq_dynamic(dGmin, V_app, mech_label)
            thetaA_Star0_dynamic = 1.0 - thetaA_H0_dynamic
            theta0_dynamic = [thetaA_Star0_dynamic, thetaA_H0_dynamic]

            def sitebal(t_local, theta):
                r_V_local, r_T_local, r_H_local = rates_r0(t_local, theta)
                if mech_label == "VT":
                    dtheta_star = (-r_V_local + 2 * r_T_local) / cmax
                    dtheta_H = (r_V_local - 2 * r_T_local) / cmax
                else:
                    dtheta_star = (r_H_local - r_V_local) / cmax
                    dtheta_H = (r_V_local - r_H_local) / cmax
                return [dtheta_star, dtheta_H]

            soln = solve_ivp(sitebal, duration, theta0_dynamic,
                             t_eval=t, max_step=maxstep, method='BDF')

            theta_at_t = soln.y
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

            GHad_range = dGmax - dGmin
            mask_min = (np.abs(GHad_t_J - dGmin) < 0.2 * GHad_range)
            mask_max = (np.abs(GHad_t_J - dGmax) < 0.2 * GHad_range)

            average_rT_dGmin = np.average(r_T_vals[mask_min])
            average_rT_dGmax = np.average(r_T_vals[mask_max])
            average_rH_dGmin = np.average(r_H_vals[mask_min])
            average_rH_dGmax = np.average(r_H_vals[mask_max])

            print(f"Average rT at {dGmin}: {average_rT_dGmin}")
            print(f"Average rT at {dGmax}: {average_rT_dGmax}")
            print(f"Average rH at {dGmin}: {average_rH_dGmin}")
            print(f"Average rH at {dGmax}: {average_rH_dGmax}")

            avg_curr_at_dGmin = np.average(np.abs(curr_dynamic[mask_min]))
            avg_curr_at_dGmax = np.average(np.abs(curr_dynamic[mask_max]))

            # Store overlay points (per mechanism)
            overlay["rT"][freq] = [
                (dGmin_dynamic, float(average_rT_dGmin)),
                (dGmax_dynamic, float(average_rT_dGmax)),
            ]
            overlay["curr"][freq] = [
                (dGmin_dynamic, float(avg_curr_at_dGmin)),
                (dGmax_dynamic, float(avg_curr_at_dGmax)),
            ]
            overlay["rH"][freq] = [
                (dGmin_dynamic, float(average_rH_dGmin)),
                (dGmax_dynamic, float(average_rH_dGmax)),
            ]

            dyn_results.append({
                "r_T": r_T_vals,
                "r_H": r_H_vals,
                "rV": r_V_vals,
                "period": 2 / freq,
                "freq": float(freq),
                "t": t.copy(),
                "curr": curr_dynamic.copy(),
                "thetaH": thetaH_array.copy(),
                "GHad_eV": GHad_t_eV.copy(),
                "Average Current": avg_curr,
                "maxstep": maxstep,
            })

            # --- OPTIONAL per-frequency plots (can comment out if too many) ---
            P_local = 2.0 / freq
            cycles_to_plot = 5
            t_start = 0.0
            t_end = cycles_to_plot * P_local
            mask = (t >= t_start) & (t <= t_end)

            # rT or rH vs time
            average_rT = np.average(r_T_vals)
            average_rH = np.average(r_H_vals)

            if mech_label == "VT":
                plt.figure(figsize=(8, 5))
                plt.plot(t[mask], r_T_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
                plt.axhline(y=average_rT, color="red", linestyle="--", linewidth=2,
                            label=f"Average rT = {average_rT:.2e}")
                plt.xlabel("Time (s)")
                plt.ylabel(r"$r_T$ (mol/cm²·s)")
                plt.title(f"r_T vs Time at {freq:.2e} Hz (VT)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(8, 5))
                plt.plot(t[mask], r_H_vals[mask], label=f"{freq:.2e} Hz", linewidth=1.8)
                plt.axhline(y=average_rH, color="red", linestyle="--", linewidth=2,
                            label=f"Average rH = {average_rH:.2e}")
                plt.xlabel("Time (s)")
                plt.ylabel(r"$r_H$ (mol/cm²·s)")
                plt.title(f"r_H vs Time at {freq:.2e} Hz (VH)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # Coverage vs time
            plt.figure(figsize=(12, 10))
            plt.plot(t[mask], thetaH_array[mask], label=f'θ_H ({freq:.2e} Hz)')
            plt.xlabel("Time (s)")
            plt.ylabel(r"$\theta_H$")
            plt.title(f'Coverage vs Time, {freq:.2e} Hz ({mech_label})')
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.show()

            # rV vs time
            plt.figure(figsize=(12, 10))
            plt.plot(t[mask], r_V_vals[mask], label=f'r_V ({freq:.2e} Hz)')
            plt.xlabel("Time (s)")
            plt.ylabel(r"$r_V$ (mol/cm²·s)")
            plt.title(f'r_V vs Time, {freq:.2e} Hz ({mech_label})')
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.show()

        dyn_results_all[mech_label] = dyn_results

###############################################################################
# STATIC VOLCANO (VT and/or VH) + OVERLAYS
###############################################################################

if do_static_volcano:

    print("\nRunning static volcano calculations...\n")

    # GHad grid for static volcano
    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 16)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor

    # Time grid
    t_end = 5
    t_eval = np.linspace(0, t_end, 200)

    static_results_VT = []
    static_results_VH = []

    def rates_r0_static(theta, GHad, mech_label):
        """
        Static rate expressions; picks k_T or k_H based on mechanism.
        """
        thetaA_star, thetaA_H = theta
        k_V_local = k_V_RDS
        if mech_label == "VT":
            k_T_local = k_T_base
            k_H_local = 0.0
        else:
            k_T_local = 0.0
            k_H_local = k_H_base

        U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F

        # Volmer
        rV = k_V_local * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
            * np.exp(beta[0] * GHad / RT) * (
                np.exp(-(beta[0]) * F * (V_app - U_V) / RT)
                - np.exp((1 - beta[0]) * F * (V_app - U_V) / RT))

        if mech_label == "VT":
            rT = k_T_local * ((thetaA_H ** 2) -
                              partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT))
            rH = 0.0
        else:
            U_H = (GHad / F) + (RT / F) * np.log(thetaA_H / thetaA_star)
            j1 = k_H_local * np.exp(-beta[1] * GHad / RT) \
                * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
            rH = j1 * (
                np.exp(-beta[1] * F * (V_app - U_H) / RT)
                - np.exp((1 - beta[1]) * F * (V_app - U_H) / RT)
            )
            rT = 0.0

        return rV, rT, rH

    def theta_H_eq_static(GHad, mech_label):
        lo, hi = 1e-9, 1 - 1e-9

        def f(thetaH):
            theta = np.array([1 - thetaH, thetaH])
            rV, rT, rH = rates_r0_static(theta, GHad, mech_label)
            if mech_label == "VT":
                return rV - 2 * rT
            else:
                return rV - rH

        sol = root_scalar(f, bracket=[lo, hi])
        return sol.root

    for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):

        # ---- VT static ----
        if "VT" in mechanisms_to_run:
            theta_H0 = theta_H_eq_static(GHad, "VT")
            theta0 = [1 - theta_H0, theta_H0]

            def ode_VT(t, th):
                rV, rT, rH = rates_r0_static(th, GHad, "VT")
                dtheta_star = (-rV + 2 * rT) / cmax
                dtheta_H = (rV - 2 * rT) / cmax
                return [dtheta_star, dtheta_H]

            sol = solve_ivp(ode_VT, (0, t_end), theta0, t_eval=t_eval, method="BDF")
            r_vals = np.array([rates_r0_static(th, GHad, "VT") for th in sol.y.T])
            r_T_vals = r_vals[:, 1]
            avg_rT = np.mean(r_T_vals[5:])

            static_results_VT.append((GHad_eV, avg_rT))
            static_summary_all.append({
                "Mechanism": "VT",
                "GHad (eV)": GHad_eV,
                "Average r_T (mol/cm²·s)": avg_rT,
                "Average rH (mol/cm²·s)": np.nan,
            })

        # ---- VH static ----
        if "VH" in mechanisms_to_run:
            theta_H0 = theta_H_eq_static(GHad, "VH")
            theta0 = [1 - theta_H0, theta_H0]

            def ode_VH(t, th):
                rV, rT, rH = rates_r0_static(th, GHad, "VH")
                dtheta_star = (rH - rV) / cmax
                dtheta_H = (rV - rH) / cmax
                return [dtheta_star, dtheta_H]

            sol = solve_ivp(ode_VH, (0, t_end), theta0, t_eval=t_eval, method="BDF")
            r_vals = np.array([rates_r0_static(th, GHad, "VH") for th in sol.y.T])
            r_H_vals = r_vals[:, 2]
            avg_rH = np.mean(r_H_vals[5:])

            static_results_VH.append((GHad_eV, avg_rH))
            static_summary_all.append({
                "Mechanism": "VH",
                "GHad (eV)": GHad_eV,
                "Average r_T (mol/cm²·s)": np.nan,
                "Average rH (mol/cm²·s)": avg_rH,
            })

    # =======================
    # PLOTTING VT STATIC + OVERLAY
    # =======================
    if "VT" in mechanisms_to_run and static_results_VT:
        GHad_vals, avg_rT_vals = zip(*static_results_VT)

        plt.figure(figsize=(10, 8))
        plt.plot(GHad_vals, avg_rT_vals, marker='o', color='blue',
                 label="Static r_T")

        overlay_VT_rT = dynamic_overlay_all["VT"]["rT"]

        for f in freq_array:
            if f not in overlay_VT_rT:
                continue
            pts = overlay_VT_rT[f]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.scatter(xs, ys, s=120, facecolors="none",
                        edgecolors="red", linewidths=2)
            for x, y in zip(xs, ys):
                plt.text(x, y, f"{f:.1e}", fontsize=14, weight="bold")

        plt.xlabel("GHad (eV)")
        plt.ylabel("Average r_T (mol/cm²·s)")
        plt.title(f"r_T vs GHad, V = {V_app} V (VT)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # =======================
    # PLOTTING VH STATIC + OVERLAY
    # =======================
    if "VH" in mechanisms_to_run and static_results_VH:
        GHad_vals, avg_rH_vals = zip(*static_results_VH)

        plt.figure(figsize=(10, 8))
        plt.plot(GHad_vals, avg_rH_vals, marker='o', color='blue',
                 label="Static r_H")

        overlay_VH_rH = dynamic_overlay_all["VH"]["rH"]

        for f in freq_array:
            if f not in overlay_VH_rH:
                continue
            pts = overlay_VH_rH[f]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.scatter(xs, ys, s=120, facecolors="none",
                        edgecolors="red", linewidths=2)
            for x, y in zip(xs, ys):
                plt.text(x, y, f"{f:.1e}", fontsize=14, weight="bold")

        plt.xlabel("GHad (eV)")
        plt.ylabel("Average r_H (mol/cm²·s)")
        plt.title(f"r_H vs GHad, V = {V_app} V (VH)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Print static summary
    if static_summary_all:
        df_summary_static = pd.DataFrame(static_summary_all)
        print("\nStatic Volcano Summary:")
        print(df_summary_static)

###############################################################################
# EXCEL EXPORT (ONE FILE)
###############################################################################

save_folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
os.makedirs(save_folder, exist_ok=True)

# Use base k-values in filename (same for both mechanisms)
output_filename = os.path.join(
    save_folder,
    make_output_filename(
        kV=k_V_RDS, kT=k_T_base, kH=k_H_base,
        freq_array=freq_array, beta=beta,
        dGmin=dGmin_dynamic, dGmax=dGmax_dynamic,
        voltage=V_app, t_switching=None
    )
)

with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
    # Dynamic sheets
    for mech_label in mechanisms_to_run:
        for res in dyn_results_all[mech_label]:
            freq_label = f"{res['freq']:.2e}Hz_{mech_label}"
            df = pd.DataFrame({
                "Time (s)": res["t"],
                "GHad (eV)": res["GHad_eV"],
                "Current (mA/cm²)": res["curr"],
                "r_T (mol/cm²·s)": res["r_T"],
                "r_V (mol/cm²·s)": res["rV"],
                "r_H (mol/cm²·s)": res["r_H"],
                "θ_H": res["thetaH"],
            })
            df.to_excel(writer, sheet_name=freq_label[:31], index=False)

    # Static summary
    if static_summary_all:
        df_static_summary = pd.DataFrame(static_summary_all)
        df_static_summary.to_excel(writer, sheet_name="Static_Summary", index=False)

print(f"\nAll results exported to Excel: {output_filename}")
