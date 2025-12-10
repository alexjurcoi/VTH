import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import altair as alt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import fsolve
plt.rcParams.update({'font.size': 14})
from tabulate import tabulate

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
AvoNum = 6.02e23  # 1/mol
partialPH2 = 1.0
beta = [0.05, 0.5]
GHad_eV = -0.15

k_V_RDS = cmax * 10**-0.68

if mechanism_choice == 0:
    k_V = k_V_RDS
    k_T = k_V * 1000
elif mechanism_choice == 1:
    k_V = k_V_RDS
    k_H = k_V * 1000

GHad = GHad_eV * AvoNum * conversion_factor  # Convert GHad from eV to J

# # potential sweep & time
UpperV = 0
LowerV = -0.2
scanrate = 0.05  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = timescan
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x):
    #timescan above is the same as single_sweep_time
    single_sweep_time = (UpperV - LowerV) / scanrate
    cycle_time = 2 * single_sweep_time

    t_in_cycle = x % cycle_time

    if t_in_cycle < single_sweep_time: #forward
        return UpperV - scanrate * t_in_cycle
    else: #reverse
        return LowerV + scanrate * (t_in_cycle - single_sweep_time)
# def potential(x):
#     if x%(2*timescan)<timescan:
#         return UpperV - scanrate*((x - timescan) % timescan)
#     else:
#         return LowerV + scanrate*(x% timescan)


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
    if mechanism_choice in [0, 2]:
        thetaStar_rate_VT = (-r_V + (2*r_T)) / cmax
        thetaH_rate_VT = (r_V - (2*r_T)) / cmax
        dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT] # [0 = star, 1 = H]
    elif mechanism_choice in [1, 3]:
        theta_star_rate = r_H-r_V      # summing all step rates based on how they affect theta_star
        theta_H_rate = r_V-r_H        # summing all step rates based on how they affect theta_H
        dthetadt = [theta_star_rate / cmax, theta_H_rate / cmax]
    return dthetadt

V = np.array([potential(ti) for ti in t])
curr1 = np.empty(len(t), dtype=object)
tcurr1= np.empty(len(t), dtype=object)

# whether a rate is positive or negative depends on the associated reaction affects each site balance
# Heyrovsky (r1) is positive in theta_star_rate because as the H reaction moves forward, the number of empty sites increases
# Volmer (r0) is negative in theta_star_rate because as the V reaction moves forward, the number of empty sites decreases
# Heyrovsky (r1) is negative in theta_H_rate because as the H reaction moves forward, the number of H-occupied sites decreases
# Volmer (r0) is positive in theta_H_rate because as the V reaction moves forward, the number of H-occupied sites increases

soln = solve_ivp(sitebal, duration, theta0, t_eval=t, method='BDF')
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
volmer_rate = r0_vals[:, 0]
curr1 = volmer_rate * -F * 1000  # mA/cm²

V_model = np.array([potential(ti) for ti in t])

########################################################################################################################
################################################ Data Import ###########################################################
########################################################################################################################
#importing all of the data files from UCSD
df_Pristine = pd.read_excel("Pristine_experimentaldata.xlsx", sheet_name= 0)

#pristine SRO BTO data
Pristine_experi_I = ((df_Pristine.iloc[562:843, 9]) / 0.188) + 0.026 #mA/cm^2
Pristine_experi_V = df_Pristine.iloc[562:843, 7]

Pristine_experi_V = np.array(Pristine_experi_V, dtype=float)
Pristine_experi_I = np.array(Pristine_experi_I, dtype=float)

# print("Pristine voltage:", Pristine_experi_V)
# print("Pristine current:", Pristine_experi_I)

#come back to SS_tot
def r_squared(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    ss_res = np.sum((data1 - data2) ** 2)
    ss_tot = np.sum((data1 - np.mean(data1)) ** 2)
    return 1 - (ss_res/ss_tot)

# Find the index of minimum voltage (most negative point)
min_V_index = np.argmin(V_model)

# Take only the values from start up to and including the minimum
V_model_masked = V_model[:min_V_index+1]
curr_model_masked = curr1[:min_V_index+1]


#pristine mask
pristine_mask = Pristine_experi_V <= 0
Pristine_experi_V_masked = Pristine_experi_V[pristine_mask]
Pristine_experi_I_masked = Pristine_experi_I[pristine_mask]

#interpolating data
interp_func = interp1d(V_model_masked, curr_model_masked, kind='linear', fill_value='extrapolate')

I_model_interp_pristine = interp_func(Pristine_experi_V_masked)
r2_pristine = r_squared(Pristine_experi_I_masked, I_model_interp_pristine)
print(f"R² for Pristine vs Model: {r2_pristine:.4f}")

table_r2 = [["Pristine Current Masked", Pristine_experi_I_masked], ["Model Current Interpolated & Masked", I_model_interp_pristine]]
print(table_r2)

## tables
table_p_mask = [["Pristine Voltage Mask (V)", Pristine_experi_V_masked],["Pristine Current (mA/cm2)", Pristine_experi_I_masked]]
table_pristine = [["Pristine Voltage (V)", Pristine_experi_V],["Pristine Current (mA/cm2)", Pristine_experi_I]]
table_model_mask = [["Model Voltage (V)", V_model_masked],["Model Current (mA/cm2)", curr_model_masked]]

# print("Pristine Data Length:", len(Pristine_experi_V), "Current Data Size:", len(Pristine_experi_I))
# print("Model Data Length:", len(V_model), "Current Data Size:", len(curr1))
print("Pristine Masked Data Length:", len(Pristine_experi_V_masked), "Current Masked Data Size:", len(Pristine_experi_I_masked))
print("Model Masked Data Length:", len(V_model_masked), "Current Masked Data Size:", len(curr_model_masked))
###########################################################################################################################
###########################################################################################################################
########################################################## PLOTS ############################################################
###########################################################################################################################
###########################################################################################################################

#standard CV plot
fig, axs = plt.subplots(figsize=(8, 10))
axs.plot(Pristine_experi_V_masked, I_model_interp_pristine, 'b', label='Model Data')
#axs.plot(Pristine_experi_V_masked, Pristine_experi_I_masked, 'r', label='Pristine Experimental Data')
axs.set_xlim(None, 0.0)
axs.set_xlabel('Voltage vs. RHE (V)')
axs.set_ylim(None, 0.1)
axs.set_ylabel('Kinetic current (mA/cm2)')
axs.set_title(r'Kinetic Current vs Voltage, Pristine Data vs Model, $k_V$ = %.2e, $beta$ = %.2f' % (k_V / cmax, beta[0]))
axs.grid()
axs.legend()
plt.show()
#
#Tafel Plot
fig, ax = plt.subplots(figsize=(8, 6))
#ax.plot(np.abs(Pt_experi_I), Pt_experi_V, 'r', label='Experimental Data')
ax.plot(np.log10(np.abs(I_model_interp_pristine)), Pristine_experi_V_masked, 'b', label='Model Data')
ax.plot(np.log10(np.abs(Pristine_experi_I_masked)), Pristine_experi_V_masked, 'r', label='Pristine Experimental Data')
ax.set_xlabel('Log Kinetic current (mA/cm2)')
ax.set_ylabel('Voltage vs. RHE (V)')
ax.set_title(r'Log Kinetic Current vs Voltage, $k_V$ = %.2e, $beta$ = %.2f' % (k_V / cmax, beta[0]))
#ax.semilogx()
ax.set_ylim(None, 0.2)
ax.grid()
ax.legend()
plt.show()

# print(table_pristine)
# print(table_model_mask)
# print(table_p_mask)

# #standard CV plot
# fig, axs = plt.subplots(figsize=(8, 10))
# axs.plot(V_model, curr1, 'b', label='Model Data')
# axs.set_xlim(None, 0.0)
# axs.set_xlabel('Voltage vs. RHE (V)')
# axs.set_ylim(None, 0.1)
# axs.set_ylabel('Kinetic current (mA/cm2)')
# axs.set_title(r'Kinetic Current vs Voltage, Pristine Data vs Model, $k_V$ = %.2e, $beta$ = %.2f' % (k_V / cmax, beta[0]))
# axs.grid()
# axs.legend()
# plt.show()
#
# #Tafel Plot
# fig, ax = plt.subplots(figsize=(8, 6))
# #ax.plot(np.abs(Pt_experi_I), Pt_experi_V, 'r', label='Experimental Data')
# ax.plot(np.abs(curr1), V_model, 'b', label='Model Data')
# ax.set_xlabel('Log Kinetic current (mA/cm2)')
# ax.set_ylabel('Voltage vs. RHE (V)')
# ax.set_title(r'Log Kinetic Current vs Voltage, $k_V$ = %.2e, $beta$ = %.2f' % (k_V / cmax, beta[0]))
# ax.semilogx()
# ax.set_ylim(None, 0.2)
# ax.grid()
# ax.legend()
# plt.show()
