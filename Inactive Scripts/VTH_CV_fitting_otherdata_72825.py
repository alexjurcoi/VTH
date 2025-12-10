import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
plt.rcParams.update({'font.size': 14})

#dictionary of all the rows in which the voltage sweeps from 0V to -0.3V in all the different H2 Sat files
scan_locations = {
    "02": {"rows": (311, 746),  "v_col": 7, "i_col": 9},
    "03": {"rows": (2137, 2463), "v_col": 1, "i_col": 2},
    "04": {"rows": (2462, 2719), "v_col": 7, "i_col": 9},
    "05": {"rows": (2402, 2752), "v_col": 7, "i_col": 9},
    "06": {"rows": (2477, 2837), "v_col": 7, "i_col": 9},
    "07": {"rows": (2569, 2941), "v_col": 7, "i_col": 9},
    "08": {"rows": (2595, 2968), "v_col": 1, "i_col": 2},
}

#id number for the scan I want to import, only valid for scans 02-08
scan_id = "05"

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
beta = [0.35, 0.5]
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
LowerV = -0.25
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

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################

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
curr_model = volmer_rate * -F * 1000  # mA/cm²

print(rf"Max and min of Curr_model at {GHad_eV}", min(curr_model), max(curr_model))

V_model = np.array([potential(ti) for ti in t])
############################################################################################################################################################
############################################################################################################################################################
########################################################## Value Extracting and Data Import ################################################################
############################################################################################################################################################
############################################################################################################################################################

loc = scan_locations[scan_id]

# Get start and stop indices
start, stop = loc["rows"]
v_col, i_col = loc["v_col"], loc["i_col"]

# Load the corresponding file
filepath = fr"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\UCSD_Data\10_CV_ScanRates_H2Sat_05_CV_C01.xlsx"
df = pd.read_excel(filepath, index_col=False)

# Extract V and I
V_exp = df.iloc[start:stop, v_col]
I_import = df.iloc[start:stop, i_col]

########################################################################################################################
################################################ Data Analysis##########################################################
########################################################################################################################

#come back to SS_tot
def r_squared(data1, data2):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    ss_res = np.sum((data1 - data2) ** 2)
    ss_tot = np.sum((data1 - np.mean(data1)) ** 2)
    return 1 - (ss_res/ss_tot)

mask_min = -0.25
mask_max = -0.01

#model mask, pristine only
model_mask = (V_model <= mask_max) & (V_model >= mask_min)
V_model_masked = V_model[model_mask]
curr_model_masked = curr_model[model_mask]

#make the mask here
scan_mask = (V_exp <= mask_max) & (V_exp >= mask_min) #insert voltage of whichever scan you want
V_exp_masked = V_exp[scan_mask]
I_import_masked = I_import[scan_mask]

I_import_masked = I_import_masked.reset_index(drop=True)
V_exp_masked = V_exp_masked.reset_index(drop=True)

adjustment_exp = 0 - (I_import_masked[0] / 0.0929)
print("Adjustment for experimental data:", adjustment_exp)
I_exp_masked = (I_import_masked / 0.0929) + adjustment_exp
print("First I_exp data point", I_import_masked[0]/0.0929)

#interpolating data
interp_func = interp1d(V_model_masked, curr_model_masked, kind='linear', fill_value='extrapolate')

#insert which scan you want here
I_model_interp = interp_func(V_exp_masked) #in the interp_func it should be the voltage mask of whatever scan you want

print("V model min and max", np.min(V_model_masked), np.max(V_model_masked))
print("V data min and max", np.min(V_exp_masked), np.max(V_exp_masked))

r2_scan = r_squared(I_exp_masked[4:], I_model_interp[4:]) #in the r^2 function it should be the current masked
print(fr"R² for V_{scan_id} vs Model: {r2_scan:.4f}")
###########################################################################################################################
###########################################################################################################################
########################################################## PLOTS ############################################################
###########################################################################################################################
###########################################################################################################################

#standard CV plot
fig, axs = plt.subplots(figsize=(8, 10))
axs.plot(V_model[4:], curr_model[4:], 'r', label='Model Data')
#axs.plot(Pt_experi_V, Pt_experi_I, 'g', label='Platinum Experimental Data')
axs.plot(V_exp_masked, I_exp_masked, 'b', label = f'H2 Saturated Data, {scan_id}')
#axs.set_xlim(None, 0)
axs.set_xlabel('Voltage vs. RHE (V)')
#axs.set_ylim(None, 0)
axs.set_ylabel('Kinetic current (mA/cm2)')
axs.set_title(fr'Kinetic Current vs Voltage, V_{scan_id} vs Model, $R²$ = {r2_scan:.4f}')
axs.grid()
axs.legend()
plt.show()

#Tafel Plot
fig, ax = plt.subplots(figsize=(8, 6))
#ax.plot(np.abs(Pt_experi_I), Pt_experi_V, 'r', label='Experimental Data')
ax.plot(np.abs(I_model_interp[4:]), V_exp_masked[4:], 'r', label='Model Data')
ax.plot(np.abs(I_exp_masked[4:]), V_exp_masked[4:], 'b', label = f'H2 Saturated Data, {scan_id}')
ax.set_xlabel('Log Kinetic currkent (mA/cm2)')
ax.set_ylabel('Voltage vs. RHE (V)')
ax.set_title(fr'Log Kinetic Current vs Voltage, V_{scan_id} vs Model, $R²$ = {r2_scan:.4f}')
ax.semilogx()
#ax.set_ylim(None, 0)
ax.grid()
ax.legend()
plt.show()

# plt.figure(figsize=(8,6))
# plt.plot(t, curr_model, label='Model Current vs Time')
# plt.xlabel("Time (s)")
# plt.ylabel("Current (mA/cm²)")
# plt.title("Current vs Time")
# plt.grid()
# plt.legend()
# plt.show()

df_compare = pd.DataFrame({
    "Experimental Current": I_exp_masked[2:],
    "Model Current": I_model_interp[2:]
})
df_compare.to_excel("r2_verification.xlsx", index=False)

# print(f"Curr_model[400] = {curr_model[400]:.3f} mA/cm² at V = {V_model[400]:.3f}")
