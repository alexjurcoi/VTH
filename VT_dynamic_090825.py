import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Ask user for mechanism choice
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

########################################  Time Function ###################################################


def make_t_eval(freq, t_switching=0.5, n_cycles=6,
                coarse_pts_per_period=40,
                halo_frac=0.05,
                halo_points=15):
    P = 1.0 / float(freq)
    t_end = t_switching + n_cycles * P

    # Coarse background grid (uniform); stop at t_end (no +0.5*dt)
    dt = P / float(coarse_pts_per_period)
    # small epsilon to include t_end
    t_coarse = np.arange(0.0, t_end + 1e-12, dt)
    t_coarse = t_coarse[t_coarse <= t_end]        # hard clip

    # Exact switch instants (clip to [0, t_end])
    switches = t_switching + np.arange(n_cycles + 1) * P
    switches = np.clip(switches, 0.0, t_end)

    # Halos around each switch (also clipped)
    hw = halo_frac * P
    halos_list = []
    for tk in switches:
        a = max(0.0, tk - hw)
        b = min(t_end, tk + hw)
        if b > a:
            halos_list.append(np.linspace(a, b, halo_points))
    halos = np.concatenate(
        halos_list) if halos_list else np.array([], dtype=float)

    # Merge, unique, and clamp again just in case
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
cmax = 7.5e-9     # mol/cm²
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = [0.443716, 0.5]
freq_array = [1000, 100, 10]

k_V_RDS = 6.590103e-05 / 5000

if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
elif mechanism_choice == 1:
    k_V = k_V_RDS
    k_H = k_V * 1000

# dG values, static volcano
dGmin_eV = -0.3  # eV
dGmax_eV = 0.3

# dG values, dynamic volcano
dGmin_dynamic = 0.05  # in eV
dGmax_dynamic = 0.14  # in eV


# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input(
    "Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input(
    "Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

# === Prepare dynamic overlay variables ===
dynamic_overlay_points = []
dynamic_overlay_by_freq = {}   # freq -> list of (GHad_eV, curr)

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

    # Initial conditions
    # Initial coverage of Hads, needs to be high as this is reduction forward
    thetaA_H0_dynamic = 0.5
    thetaA_Star0_dynamic = 1.0 - thetaA_H0_dynamic  # Initial coverage of empty sites
    theta0_dynamic = [thetaA_Star0_dynamic, thetaA_H0_dynamic]

    for freq in freq_array:
        print(f"\nRunning simulation with period = {freq:.2e} Hz...")

        # time spacing
        t_switching = 0.5
        t, max_time = make_t_eval(freq, t_switching=0.5, n_cycles=100,
                                  coarse_pts_per_period=40, halo_frac=0.05, halo_points=100)
        duration = [t[0], t[-1]]

        # keep the solver from skipping over switch neighborhoods
        P = 1.0 / freq
        maxstep = P / 50.0  # simple, safe choice

        print("Max Step: ", maxstep)

        duration = [0, max_time]
        time_index = [t]

        # function for defining how dGmin and dGmax are applied to the model
        def dGvt(t):
            if t < t_switching:
                return dGmin
            else:
                return dGmin if int((t - t_switching) * freq) % 2 == 0 else dGmax

        # setting potential for static hold
        def potential(t): return -0.1

        # equil
        def eqpot(theta, GHad):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # unpack surface coverage

            # Volmer
            U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            # U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)

            # Heyrovsky
            U_H = 0
            if mechanism_choice == 0:
                U_11 = GHad / F
                U_12 = (RT / F) * np.log(thetaA_H / thetaA_star)
                U_H = U_11 + U_12

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

        print("Solver length: ", len(soln.t))

        GHad_t_J = np.array([dGvt(time) for time in t])
        GHad_t_eV = GHad_t_J / (Avo * conversion_factor)

        r0_vals = np.array([rates_r0(time, theta)
                           for time, theta in zip(t, theta_at_t.T)])
        r_V_vals = r0_vals[:, 0]
        r_T_vals = r0_vals[:, 1]
        curr_dynamic = r_V_vals * -F * 1000  # mA/cm²

        avg_curr = np.abs(np.average(curr_dynamic))
        avg_currents.append(avg_curr)

        print(f"Average Dynamic Current for {freq} = ", avg_curr)

        # Mask for each dynamic GHad value
        mask_min=np.isclose(GHad_t_J, dGmin, rtol=1e-8, atol=1e-20)
        mask_max=np.isclose(GHad_t_J, dGmax, rtol=1e-8, atol=1e-20)

        # Absolute value of the minimum current (max magnitude)
        avg_curr_at_dGmin=np.average(np.abs(curr_dynamic[mask_min]))
        avg_curr_at_dGmax=np.average(np.abs(curr_dynamic[mask_max]))

        avg_currents_dGmin.append(avg_curr_at_dGmin)
        avg_currents_dGmax.append(avg_curr_at_dGmax)

        # Save them for overlay plotting
        dynamic_overlay_points.append((dGmin_dynamic, avg_curr_at_dGmin))
        dynamic_overlay_points.append((dGmax_dynamic, avg_curr_at_dGmax))


        # ... inside: for freq in freq_array:
        # Save per-freq overlay points (in eV for the x-axis of the volcano plot)
        dynamic_overlay_by_freq[freq]=[
            (dGmin_dynamic, avg_curr_at_dGmin),
            (dGmax_dynamic, avg_curr_at_dGmax),
]

# =============================================================================
#         print(f"Max |Current| at GHad = {dGmin_dynamic:.2f} eV: {avg_curr_at_dGmin:.3f} mA/cm²")
#         print(f"Max |Current| at GHad = {dGmax_dynamic:.2f} eV: {avg_curr_at_dGmax:.3f} mA/cm²")
# =============================================================================

        dyn_results.append({
            "r_T": r_T_vals,
            "period": 1 / (freq),
            "freq": float(freq),
            "t": t.copy(),
            "curr": np.abs(curr_dynamic.copy()),
            "thetaH": thetaH_array.copy(),
            "GHad_eV": GHad_t_eV.copy(),
            "Average Current": avg_curr,
            # "curr_at_dGmin": avg_curr_at_dGmin,
            # "curr_at_dGmax": avg_curr_at_dGmax,
        })

    freq_labels=[f"{res['freq']:.2e}" for res in dyn_results]
# =============================================================================
#     curr_at_dGmin = [res["curr_at_dGmin"] for res in dyn_results]
#     curr_at_dGmax = [res["curr_at_dGmax"] for res in dyn_results]
# =============================================================================

    plt.figure(figsize=(8, 5))
    plt.scatter(freq_labels, avg_currents_dGmin, color="blue", marker="o",
                label=f"Avg |Current| at {dGmin_dynamic:.2f} eV")
    plt.scatter(freq_labels, avg_currents_dGmax, color="red", marker="x",
                label=f"Avg |Current| at {dGmax_dynamic:.2f} eV")  # Using 'x' for better visual distinction
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Current (mA/cm²)")
    plt.title("Dynamic GHad(t): Average Current Per Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(t[100:1000], r_T_vals[100:1000])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("TOF (mol/cm² s)")
    plt.show()

    # plt.figure(figsize=(10, 6))
    #
    # plt.scatter(freq_labels, curr_at_dGmin, color="blue", marker="o",
    #             label=f"Max |Current| at {dGmin_dynamic:.2f} eV")
    # plt.scatter(freq_labels, curr_at_dGmax, color="red", marker="x",
    #             label=f"Max |Current| at {dGmax_dynamic:.2f} eV")
    #
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Current (mA/cm²)")
    # plt.title("Dynamic GHad(t): Max & Min Current for Each Frequency")
    # plt.grid(axis="y", linestyle=":")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # === Plot Current and r_T vs Time for Each Frequency in Subplots ===
    n_freqs=len(dyn_results)
    fig, axes=plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes=axes.flatten()  # flatten 2D array to 1D for easy iteration


    for i, res in enumerate(dyn_results):
        mask=res["t"] >= 0.5
        ax=axes[i]
        label=f"P={res['period']:.1e}s (f={res['freq']:.2e} Hz)"
        ax.plot(res["t"][mask], res["curr"][mask],
                color="b", label='Current (from r_V)')
        ax2=ax.twinx()

        ax2.plot(res["t"][mask], res["r_T"][mask], color='r', label='r_T')
        ax.set_title(label, fontsize=10)
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (mA/cm²)")
        ax.set_xlim(res["t"][mask][0], res["t"][mask][-1])


        # Combine legends from both axes
        lines1, labels1=ax.get_legend_handles_labels()
        lines2, labels2=ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Hide any unused subplots (if n_freqs < 6)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # === Plot Current and ThetaH vs Time for Each Frequency in Subplots ===
    n_freqs=len(dyn_results)
    fig2, axes2=plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes2=axes2.flatten()  # flatten 2D array to 1D for easy iteration

    for i, res in enumerate(dyn_results):
        mask=res["t"] >= 0.5
        ax=axes2[i]
        label=f"P={res['period']:.1e}s (f={res['freq']:.2e} Hz)"
        ax.plot(res["t"][mask], res["curr"][mask],
                color="b", label='Current (from r_V)')
        ax2=ax.twinx()

        ax2.plot(res["t"][mask], res["thetaH"][mask],
                 color="g", label='θ_H (fraction)')
        ax.set_title(label, fontsize=10)
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (mA/cm²)")
        ax.set_xlim(res["t"][mask][0], res["t"][mask][-1])

        # Combine legends from both axes
        lines1, labels1=ax.get_legend_handles_labels()
        lines2, labels2=ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Hide any unused subplots (if n_freqs < 6)
    for j in range(i + 1, len(axes2)):
        fig2.delaxes(axes2[j])

    # === Additional plot: Half-data view for better detail ===
    fig_half, axes_half=plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True)
    axes_half=axes_half.flatten()

    for i, res in enumerate(dyn_results[:4]):
        ax=axes_half[i]
        t_arr=res["t"]
        curr_arr=res["curr"]             # Current (from r_V) in mA/cm²
        r_T_arr=res["r_T"]
        thetaH_arr=res["thetaH"]
        freq=res["freq"]
        period=1 / freq

        # Zoom window: 3 cycles after switching
        t_start=0.5
        t_end=t_start + 6 * period
        mask=(t_arr >= t_start) & (t_arr <= t_end)

        t_zoom=t_arr[mask]
        curr_zoom=curr_arr[mask]
        r_T_zoom=r_T_arr[mask]
        thetaH_zoom=thetaH_arr[mask]

        # Left axis: current
        ax.plot(t_zoom, curr_zoom, color='blue',
                lw=1.5, label="Current (from r_V)")
        ax.set_ylabel("Current (mA/cm²)", color="blue")
        ax.tick_params(axis='y', labelcolor="blue")

        # Right axis: r_T and θ_H
        ax2=ax.twinx()
        ax2.plot(t_zoom, thetaH_zoom, color='green', lw=1.5, label="θ_H")
        ax2.set_ylabel("θ_H (fraction)", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Titles & formatting
        ax.set_title(f"P={period:.1e}s (f={freq:.2e} Hz)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.set_xlim(t_start, t_end)

        # Combine legends from both axes
        lines, labels=ax.get_legend_handles_labels()
        lines2, labels2=ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")

    # === Additional plot: Half-data view for better detail ===
    fig_half2, axes_half2=plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True)
    axes_half2=axes_half2.flatten()

    for i, res in enumerate(dyn_results[:4]):
        ax=axes_half2[i]
        t_arr=res["t"]
        curr_arr=res["curr"]             # Current (from r_V) in mA/cm²
        r_T_arr=res["r_T"]
        thetaH_arr=res["thetaH"]
        freq=res["freq"]
        period=1 / freq

        # Zoom window: 3 cycles after switching
        t_start=0.5
        t_end=t_start + 6 * period
        mask=(t_arr >= t_start) & (t_arr <= t_end)

        t_zoom=t_arr[mask]
        curr_zoom=curr_arr[mask]
        r_T_zoom=r_T_arr[mask]
        thetaH_zoom=thetaH_arr[mask]

        # Left axis: current
        ax.plot(t_zoom, curr_zoom, color='blue',
                lw=1.5, label="Current (from r_V)")
        ax.set_ylabel("Current (mA/cm²)", color="blue")
        ax.tick_params(axis='y', labelcolor="blue")

        # Right axis: r_T and θ_H
        ax2=ax.twinx()
        ax2.plot(t_zoom, r_T_zoom, color='red', lw=1.5, label="r_T")
        ax2.set_ylabel("r_T (mol/cm²·s)", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Titles & formatting
        ax.set_title(f"P={period:.1e}s (f={freq:.2e} Hz)", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.set_xlim(t_start, t_end)

        # Combine legends from both axes
        lines, labels=ax.get_legend_handles_labels()
        lines2, labels2=ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")

    # Optional: adjust overall title
    fig_half.suptitle(
        "Zoomed View: Second Half of Current Data for Each Frequency", fontsize=14)
    plt.show()

    # Create a DataFrame
    data={
        "Time (s)": t,
# =============================================================================
#         "r_V_vals": r_V_vals,
#         "r_T_vals": r_T_vals,
#         "Theta_H": thetaH_array,
#         "GHad (eV)": GHad_t_eV,
# =============================================================================
        "Current (mA/cm²)": curr_dynamic
    }
    df=pd.DataFrame(data)

    # Save to Excel
    output_filename="dynamic_simulation_output.xlsx"
    df.to_excel(output_filename, index=False)

    print(f"\n Results exported to Excel: {output_filename}")

# === STATIC VOLCANO PLOT ===
if do_static_volcano:
    print("\nRunning static volcano plot...")

    # Initial conditions
    # Initial coverage of Hads, needs to be high as this is reduction forward
    thetaA_H0_static=0.5
    thetaA_Star0_static=1.0 - thetaA_H0_static  # Initial coverage of empty sites
    theta0_static=[thetaA_Star0_static, thetaA_H0_static]

    GHad_eV_list=np.linspace(dGmin_eV, dGmax_eV, 25)
    GHad_J_list=GHad_eV_list * Avo * conversion_factor
    GHad_results=[]

# Increase the number of evaluation points for a finer resolution
    t_end=1000  # a large value to ensure steady-state
    # Increased resolution to 10,000 points
    t_eval=np.linspace(0, t_end, 10000)

    for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):
        def potential_static(t): return -0.1

        def eqpot_static(theta):
            theta=np.asarray(theta)
            thetaA_star, thetaA_H=theta  # unpack surface coverage

            # Volmer
            U_V=0
            U_V=(-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            # U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)

            # Heyrovsky
            U_H=0
            if mechanism_choice == 1:
                U_11=GHad / F
                U_12=(RT / F) * np.log(thetaA_H / thetaA_star)
                U_H=U_11 + U_12

            return U_V, U_H


        def rates_r0_static(t_static, theta):
            theta=np.asarray(theta)
            thetaA_star, thetaA_H=theta  # surface coverages again, acting as concentrations
            V=potential_static(t_static)  # Use t directly (scalar)
            # call function to find U for given theta
            U_V, U_H=eqpot_static(theta)

            # Volmer Rate Equation
            r_V=k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * (
                        np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

            # Tafel Rate Equation
            r_T=0
            if mechanism_choice == 0:
                r_T=k_T * ((thetaA_H ** 2) - (partialPH2 *
                           (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))

            # Heyrovsky Rate Equation
            r_H=0
            if mechanism_choice == 1:
                j1=k_H * np.exp(-beta[1] * GHad / RT) * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                exp21=np.exp(-beta[1] * F * (V - U_H) / RT)
                exp22=np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                r_H=j1 * (exp21 - exp22)

            return r_V, r_T, r_H


        def sitebal_static(t_static, theta):
            r_V, r_T, r_H=rates_r0_static(t_static, theta)
            if mechanism_choice in [0]:
                thetaStar_rate_VT=(-r_V + (2 * r_T)) / cmax
                thetaH_rate_VT=(r_V - (2 * r_T)) / cmax
                # [0 = star, 1 = H]
                dthetadt=[(thetaStar_rate_VT), thetaH_rate_VT]
            elif mechanism_choice in [1]:
                theta_star_rate=r_H - r_V  # summing all step rates based on how they affect theta_star
                theta_H_rate=r_V - r_H  # summing all step rates based on how they affect theta_H
                dthetadt=[theta_star_rate / cmax, theta_H_rate / cmax]
            return dthetadt


        soln=solve_ivp(sitebal_static, (0, t_end), theta0_static,
                       t_eval=t_eval, method='BDF', max_step=0.10)
        r0_vals=np.array([rates_r0_static(time, theta)
                         for time, theta in zip(t_eval, soln.y.T)])
        curr_static=r0_vals[:, 0] * -F * 1000  # mA/cm²
        average_current=np.average(curr_static)
        GHad_results.append((GHad_eV, average_current))

# =============================================================================
#     print(f"Curr_model[400] = {curr_static[400]:.3f} mA/cm²")
# =============================================================================

    # Volcano plot
    GHad_vals, abs_currents=zip(*GHad_results)
    plt.figure(figsize=(8, 5))
    plt.plot(GHad_vals, abs_currents, label='Static GHad Scan')

    if dynamic_overlay_by_freq:
        # choose a unique marker & color per frequency
        freq_markers={freq_array[0]: "D", freq_array[1]
            : "^", freq_array[2]: "s"}     # change as you like
        freq_colors={freq_array[0]: "red", freq_array[1]
            : "tab:orange", freq_array[2]: "tab:green"}

        for f in freq_array:
            if f not in dynamic_overlay_by_freq:
                continue
            pts=dynamic_overlay_by_freq[f]
            xs=[p[0] for p in pts]  # GHad (eV)
            ys=[p[1] for p in pts]  # current

            # one scatter call per frequency -> one legend entry per frequency
            plt.scatter(
                xs, ys,
                marker=freq_markers.get(f, "x"),
                s=120,
                facecolors="none",            # hollow markers; remove if you want filled
                edgecolors=freq_colors.get(f, None),
                linewidths=2,
                label=f"Dynamic (f={f} Hz)",
                zorder=3,
            )

        plt.legend()


    plt.xlabel("GHad (eV)")
    plt.ylabel("Max |Current Density| (mA/cm²)")
    plt.title(
        f"Volcano Plot: Max Current vs GHad, $k_V$ ={k_V / cmax:.2e}, $beta$ = {beta[0]}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    max_index=np.argmax(abs_currents)
    print(
        f"\nMax Current (static): {abs_currents[max_index]:.3f} mA/cm² at GHad = {GHad_vals[max_index]:.3f} eV")
    print("\nStatic Volcano Summary:")
    for g, c in GHad_results:
        print(f"GHad = {g:.3f} eV → Max |Current| = {c:.3f} mA/cm²")
