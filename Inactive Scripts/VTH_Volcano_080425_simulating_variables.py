import numpy as np
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
GHad_eV = -0.30
period = 1 #seconds

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
dGmin_dynamic = 0.10  # in eV
dGmax_dynamic = 0.15  # in eV

max_time = 20  # seconds
interval_factor = 100
t = np.linspace(0.0, max_time, int(max_time / (period / interval_factor)) + 1)
t_switching = 5
duration = [0, max_time]
time_index = [t]

# Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
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

    #function for defining how dGmin and dGmax are applied to the model
    def dGvt(t):
        if t < t_switching:
            return dGmin
        else:
            return dGmin if ((t - t_switching) // period) % 2 == 0 else dGmax


    #setting potential for static hold
    def potential(t): return -0.1

    #solve for initial coverage after each iteration of solve_ivp
    def equil_theta(GHad, t):
        V = potential(0)
        thetaH_equil = 1 / (1 + np.exp((F * V + GHad) / RT))
        return thetaH_equil

    #equil potential
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


    T1_index.clear()
    T2_index.clear()

    soln = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF', dense_output=True)
    theta_at_t = soln.sol(t)  # shape: (2, len(t))
    r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, theta_at_t.T)])
    r_V_vals = r0_vals[:, 0]
    r_T_vals = r0_vals[:, 1]
    curr_dynamic = r_V_vals * -F * 1000  # mA/cm²
    GHad_t_J = np.array([dGvt(time) for time in t])
    GHad_t_eV = GHad_t_J / (Avo * conversion_factor)


    ##### equilibrium section#######
    def thetaH_equilibrium(GHad_array):
        GHad_array = np.asarray(GHad_array)  # Ensure it's an array
        V = potential(0)  # constant voltage
        return 1 / (1 + np.exp((F * V + GHad_array) / RT))

    thetaH_equil_array = thetaH_equilibrium(GHad_t_J)

    plt.plot(t[4000:6000], theta_at_t[1, 4000:6000], label="ODE theta_H")
    plt.plot(t[4000:6000], thetaH_equil_array[4000:6000], '--', label="Equilibrium theta_H")
    plt.legend()

    thetaH_array = theta_at_t[1, :]  # thetaA_H values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, curr_dynamic, label='Volmer Current')
    plt.ylabel("Current Density (mA/cm²)")
    plt.title("Dynamic GHad(t): Current vs Time, $k_V$ = {:.2e}, $k_T$ = {:.2e}, period = {:.2e}".format(k_V / cmax, k_T / cmax, period))
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, GHad_t_eV)
    plt.ylabel("GHad (eV)")
    plt.xlabel("Time (s)")
    plt.title("Dynamic GHad(t): GHad vs Time")
    plt.tight_layout()

    print("Length of theta H array: ", len(thetaH_array))

    plt.subplot(3, 1, 3)
    plt.ylabel("Coverage (Theta H)")
    plt.xlabel("Time (s)")
    plt.title("Dynamic GHad(t): Coverage vs Time")
    plt.plot(t, thetaH_array, label='Theta H', color="g")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(t[1:], r_V_vals[1:], label='r_V')
    plt.plot(t[1:], r_T_vals[1:], label='r_T')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(T1_index[100:2000], label='T1')
    plt.plot(T2_index[100:2000], label='T2')
    plt.plot(r_T_vals[100:2000], label='r_T')
    plt.xlabel("Evaluation index (arbitrary units)")
    plt.ylabel("Value")
    plt.title(rf"Sequential Evaluation of T1, T2, and r_T, period = {period} seconds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(thetaH_array)

    # Mask-based max current extraction
    mask_min = np.isclose(GHad_t_eV, dGmin_dynamic)
    mask_max = np.isclose(GHad_t_eV, dGmax_dynamic)
    max_curr_min = np.max(np.abs(curr_dynamic[mask_min]))
    max_curr_max = np.max(np.abs(curr_dynamic[mask_max]))
    dynamic_overlay_points.append((dGmin_dynamic, max_curr_min))
    dynamic_overlay_points.append((dGmax_dynamic, max_curr_max))

    print(f"\nMax |Current| at GHad = {dGmin_eV:.2f} eV: {max_curr_min:.3f} mA/cm²")
    print(f"Max |Current| at GHad = {dGmax_eV:.2f} eV: {max_curr_max:.3f} mA/cm²")

# === STATIC VOLCANO PLOT ===
if do_static_volcano:
    print("\nRunning static volcano plot...")
    beta_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    volcano_data = []
    GHad_eV_list = np.linspace(dGmin_eV, dGmax_eV, 25)
    GHad_J_list = GHad_eV_list * Avo * conversion_factor
    GHad_results = []

    for b in beta_list:
        print(f"\nComputing volcano for β = {b:.2f}...")
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
                r_V = k_V * (thetaA_star ** (1 - b)) * (thetaA_H ** b) * np.exp(b * GHad / RT) * (
                        np.exp(-b * F * (V - U_V) / RT) - np.exp((1 - b) * F * (V - U_V) / RT))

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

            volcano_data.append((b, GHad_results))

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

    cmap = plt.get_cmap('rainbow')
    n_colors = len(volcano_data)

    for i, (b, results) in enumerate(volcano_data):
        GHad_vals, abs_currents = zip(*results)
        color = cmap(i / (n_colors - 1))  # evenly spaced from 0 to 1
        plt.plot(GHad_vals, abs_currents, marker='o', color=color, label=f"β = {b:.2f}")

    # if dynamic_overlay_points:
    #     for ghad_val, curr_val in dynamic_overlay_points:
    #         plt.scatter([ghad_val], [curr_val], color='red', marker='x', s=100,
    #                     label=f'Dynamic Max @ {ghad_val:.2f} eV, {curr_val:.2f} mA/cm²')

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.xlabel("GHad (eV)")
    plt.ylabel("Max |Current Density| (mA/cm²)")
    plt.title("Volcano Plot: Max Current vs GHad for Multiple β Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
