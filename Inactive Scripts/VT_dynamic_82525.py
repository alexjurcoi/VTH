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
beta = [0.35, 0.5]
freq_array = [1, 10, 100, 1000]

k_V_RDS = cmax * 10**3.7

if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
elif mechanism_choice == 1:
    k_V = k_V_RDS
    k_H = k_V * 1000

## dG values, static volcano
dGmin_eV = -0.3  # eV
dGmax_eV = 0.3

# dG values, dynamic volcano
dGmin_dynamic = 0.05 # in eV
dGmax_dynamic = 0.14  # in eV

# Initial conditions
thetaA_H0 = 0.5  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = [thetaA_Star0, thetaA_H0]

# === Prompt User ===
print("Choose which simulations to run:")
do_static_volcano = input("Run static volcano plot? (y/n): ").strip().lower() == 'y'
do_dynamic_ghad = input("Run dynamic GHad(t) simulation? (y/n): ").strip().lower() == 'y'

# === Prepare dynamic overlay variables ===
dynamic_overlay_points = []

# === DYNAMIC GHad(t) SIMULATION ===
if do_dynamic_ghad:
    T1_index = []
    T2_index = []
    print("\nRunning dynamic GHad(t) simulation...")
    thetaH_array = []
    # Time-varying GHad values (in J)
    dGmin = dGmin_dynamic * Avo * conversion_factor
    dGmax = dGmax_dynamic * Avo * conversion_factor

    dyn_results = []

    for freq in freq_array:
        print(f"\nRunning simulation with period = {freq:.2e} Hz...")

        # time spacing
        t_switching = 0.5  # no dynamic switching before this time
        n_cycle = 20
        max_time = (n_cycle / freq) + t_switching # cycles
        base_points = 200
        n_points = int(base_points * n_cycle * np.sqrt(freq))
        t = np.linspace(0, max_time, num=int(n_points))

        if freq < k_V_RDS:
            maxstep = k_V_RDS / 100
        else:
            maxstep = 1 / (100 * freq)


        print("Max Step: ", maxstep)

        duration = [0, max_time]
        time_index = [t]

        #function for defining how dGmin and dGmax are applied to the model
        def dGvt(t):
            if t < t_switching:
                return dGmin
            else:
                return dGmin if int((t - t_switching) * freq) % 2 == 0 else dGmax

        #setting potential for static hold
        def potential(t): return -0.1

        #equil
        def eqpot(theta, GHad):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # unpack surface coverage

            ##Volmer
            U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            # U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)

            ##Heyrovsky
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
            U_V, U_H = eqpot(theta, GHad)  # call function to find U for given theta

            ##Volmer Rate Equation
            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * (
                        np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

            r_T = 0
            if mechanism_choice == 0:
                T_1 = (thetaA_H ** 2)
                T_2 = (partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT))
                r_T = k_T * (T_1 - T_2)

                #r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))
            ##Heyrovsky Rate Equation
            r_H = 0
            if mechanism_choice == 1:
                j1 = k_H * np.exp(-beta[1] * GHad / RT) * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
                exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                r_H = j1 * (exp21 - exp22)

            # T1_index.append(T_1)
            # T2_index.append(T_2)
            return r_V, r_T, r_H


        def sitebal(t, theta):
            r_V, r_T, r_H = rates_r0(t, theta)
            if mechanism_choice in [0]:
                thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
                thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
                dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT]  # [0 = star, 1 = H]
            elif mechanism_choice in [1]:
                theta_star_rate = r_H - r_V  # summing all step rates based on how they affect theta_star
                theta_H_rate = r_V - r_H  # summing all step rates based on how they affect theta_H
                dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
            return dthetadt

        soln = solve_ivp(sitebal, duration, theta0, t_eval = t, method ='BDF', max_step = maxstep)
        theta_at_t = soln.y  # shape: (2, len(t))
        thetaH_array = theta_at_t[:, 1]
        
        print("Solver length: ", len(soln.t))
        
        GHad_t_J = np.array([dGvt(time) for time in t])
        GHad_t_eV = GHad_t_J / (Avo * conversion_factor)

        r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, theta_at_t.T)])
        r_V_vals = r0_vals[:, 0]
        r_T_vals = r0_vals[:, 1]
        curr_dynamic = r_V_vals * -F * 1000  # mA/cm²

        avg_curr = np.abs(np.average(curr_dynamic))
        
        print(f"Average Dynamic Current for {freq} = ", avg_curr)

        # # Mask for each dynamic GHad value
        # mask_min = np.isclose(GHad_t_J, dGmin, rtol=1e-8, atol=1e-20)
        # mask_max = np.isclose(GHad_t_J, dGmax, rtol=1e-8, atol=1e-20)
        #
        # # Absolute value of the minimum current (max magnitude)
        # max_curr_at_dGmin = np.max(np.abs(curr_dynamic[mask_min]))
        # max_curr_at_dGmax = np.max(np.abs(curr_dynamic[mask_max]))
        #
        # # Save them for overlay plotting
        # dynamic_overlay_points.append((dGmin_dynamic, max_curr_at_dGmin))
        # dynamic_overlay_points.append((dGmax_dynamic, max_curr_at_dGmax))
        #
        # print(f"Max |Current| at GHad = {dGmin_dynamic:.2f} eV: {max_curr_at_dGmin:.3f} mA/cm²")
        # print(f"Max |Current| at GHad = {dGmax_dynamic:.2f} eV: {max_curr_at_dGmax:.3f} mA/cm²")

        dyn_results.append({
            "period": 1 / (freq),
            "freq": float(freq),
            "t": t.copy(),
            "curr": np.abs(curr_dynamic.copy()),
            "thetaH": thetaH_array.copy(),
            "GHad_eV": GHad_t_eV.copy(),
            "Average Current": avg_curr,
            # "curr_at_dGmin": max_curr_at_dGmin,
            # "curr_at_dGmax": max_curr_at_dGmax,
        })

    freq_labels = [f"{res['freq']:.2e}" for res in dyn_results]
# =============================================================================
#     curr_at_dGmin = [res["curr_at_dGmin"] for res in dyn_results]
#     curr_at_dGmax = [res["curr_at_dGmax"] for res in dyn_results]
# =============================================================================

    plt.bar(freq_labels, avg_curr, color="blue", label=f"Max |Current| at {dGmin_dynamic:.2f} eV")
    plt.xlim(10000, None)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Current (mA/cm²")
    plt.title("Dynamic GHad(t): Average Current Per Frequency")
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

    # === Plot Current vs Time for Each Frequency in Subplots ===
    n_freqs = len(dyn_results)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()  # flatten 2D array to 1D for easy iteration


    for i, res in enumerate(dyn_results):
        mask = res["t"] >= 0.5
        ax = axes[i]
        label = f"P={res['period']:.1e}s (f={res['freq']:.2e} Hz)"
        ax.plot(res["t"][mask], res["curr"][mask], color="b")
        ax.set_title(label, fontsize=10)
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (mA/cm²)")
        ax.set_xlim(res["t"][mask][0], res["t"][mask][-1])

    # Hide any unused subplots (if n_freqs < 6)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

# =============================================================================
#     fig.suptitle(f"Dynamic GHad(t): Current vs Time", fontsize=16)
#     plt.subplots_adjust(top=0.90)
#     plt.show()
# =============================================================================

    # plt.figure(figsize=(12, 6))
    # plt.plot(t, thetaH_array, label='Theta H')
    # plt.title("Rate of change of Theta_H over time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("d(Theta_H)/dt")
    # plt.grid()
    # plt.show()

    # cut = 5
    # # Plot
    # plt.figure(figsize=(16, 12))
    # plt.subplot(4, 1, 1)
    # plt.plot(t, curr_dynamic, label='Volmer Current', marker = "o")
    # plt.ylabel("Current Density (mA/cm²)")
    # plt.title("Dynamic GHad(t): Current vs Time, period = {:.2e}".format(freq))
    # plt.legend()
    #
    # plt.subplot(4, 1, 2)
    # plt.plot(t, GHad_t_eV, marker= 'o')
    # plt.ylabel("GHad (eV)")
    # plt.xlabel("Time (s)")
    # plt.title("Dynamic GHad(t): GHad vs Time")
    # plt.tight_layout()
    #
    # plt.subplot(4, 1, 3)
    # plt.ylabel("Coverage (Theta H)")
    # plt.xlabel("Time (s)")
    # plt.title("Dynamic GHad(t): Coverage vs Time")
    # plt.ylim(0, 0.4)
    # plt.plot(t, thetaH_array, label='Theta H', color="g", marker='o')
    #
    # plt.subplot(4, 1, 4)
    # plt.ylabel('Time')
    # plt.xlabel('Index')
    # plt.title("Time vs Index")
    # plt.plot(t, marker='o')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Assuming t is already defined
    # plt.figure(figsize=(8, 4))
    # plt.plot(t, marker='o')
    # plt.xlabel("Index")
    # plt.ylabel("Time (s)")
    # plt.title("Time Array (t)")
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(12,6))
    # plt.plot(t[1:], r_V_vals[1:], label='r_V')
    # plt.plot(t[1:], r_T_vals[1:], label='r_T')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(T1_index[100:2000], label='T1')
    # plt.plot(T2_index[100:2000], label='T2')
    # plt.plot(r_T_vals[100:2000], label='r_T')
    # plt.xlabel("Evaluation index (arbitrary units)")
    # plt.ylabel("Value")
    # plt.title(rf"Sequential Evaluation of T1, T2, and r_T, period = {period} seconds")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

# =============================================================================
#     print(f"\nMax |Current| at GHad = {dGmin_eV:.2f} eV: {max_curr_at_dGmin:.3f} mA/cm²")
#     print(f"Max |Current| at GHad = {dGmax_eV:.2f} eV: {max_curr_at_dGmax:.3f} mA/cm²")
# =============================================================================

    # Create a DataFrame
    data = {
        "Time (s)": t,
# =============================================================================
#         "r_V_vals": r_V_vals,
#         "r_T_vals": r_T_vals,
#         "Theta_H": thetaH_array,
#         "GHad (eV)": GHad_t_eV,
# =============================================================================
        "Current (mA/cm²)": curr_dynamic
    }
    df = pd.DataFrame(data)
    
    # Save to Excel
    output_filename = "dynamic_simulation_output.xlsx"
    df.to_excel(output_filename, index=False)
    
    print(f"\n Results exported to Excel: {output_filename}")

# === STATIC VOLCANO PLOT ===
if do_static_volcano:
    print("\nRunning static volcano plot...")

    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 25)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor
    GHad_results = []

    for GHad, GHad_eV in zip(GHad_J_list, GHad_eV_list):
        def potential(t): return -0.1


        def eqpot(theta):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # unpack surface coverage

            ##Volmer
            U_V = 0
            U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
            # U relies on the free energy of hydrogen adsorption plus the log of surface coverage (considered a concentration)

            ##Heyrovsky
            U_H = 0
            if mechanism_choice == 1:
                U_11 = GHad / F
                U_12 = (RT / F) * np.log(thetaA_H / thetaA_star)
                U_H = U_11 + U_12

            return U_V, U_H


        def rates_r0(t, theta):
            theta = np.asarray(theta)
            thetaA_star, thetaA_H = theta  # surface coverages again, acting as concentrations
            V = potential(t)  # Use t directly (scalar)
            U_V, U_H = eqpot(theta)  # call function to find U for given theta

            ##Volmer Rate Equation
            r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) * np.exp(beta[0] * GHad / RT) * (
                        np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))

            ##Tafel Rate Equation
            r_T = 0
            if mechanism_choice == 0:
                r_T = k_T * ((thetaA_H ** 2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2 * GHad) / RT)))

            ##Heyrovsky Rate Equation
            r_H = 0
            if mechanism_choice == 1:
                j1 = k_H * np.exp(-beta[1] * GHad / RT) * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
                exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
                exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
                r_H = j1 * (exp21 - exp22)

            return r_V, r_T, r_H


        def sitebal(t, theta):
            r_V, r_T, r_H = rates_r0(t, theta)
            if mechanism_choice in [0, 2]:
                thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
                thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
                dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT]  # [0 = star, 1 = H]
            elif mechanism_choice in [1, 3]:
                theta_star_rate = r_H - r_V  # summing all step rates based on how they affect theta_star
                theta_H_rate = r_V - r_H  # summing all step rates based on how they affect theta_H
                dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
            return dthetadt


        soln = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
        r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
        curr_static = r0_vals[:, 0] * -F * 1000  # mA/cm²
        max_current = np.abs(curr_static[100])
        GHad_results.append((GHad_eV, max_current))

    print(f"Curr_model[400] = {curr_static[400]:.3f} mA/cm²")

    # Volcano plot
    GHad_vals, abs_currents = zip(*GHad_results)
    plt.figure(figsize=(8, 5))
    plt.plot(GHad_vals, abs_currents, marker='o', label='Static GHad Scan')

    if dynamic_overlay_points:
        for ghad_val, curr_val in dynamic_overlay_points:
            plt.scatter([ghad_val], [curr_val], color='red', marker='x', s=100,
                        label=f'Dynamic Max @ {ghad_val:.2f} eV and {curr_val:.2f} mA/cm²')
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())

    plt.xlabel("GHad (eV)")
    plt.ylabel("Max |Current Density| (mA/cm²)")
    plt.title(
        f"Volcano Plot: Max Current vs GHad, $k_V$ ={k_V / cmax:.2e}, $k_T$ = {k_T / cmax:.2e}, $beta$ = {beta[0]}, $V$ = {potential(t)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    max_index = np.argmax(abs_currents)
    print(f"\nMax Current (static): {abs_currents[max_index]:.3f} mA/cm² at GHad = {GHad_vals[max_index]:.3f} eV")
    print("\nStatic Volcano Summary:")
    for g, c in GHad_results:
        print(f"GHad = {g:.3f} eV → Max |Current| = {c:.3f} mA/cm²")
