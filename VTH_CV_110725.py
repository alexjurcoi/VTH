import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 14})

###########################################################################################################################
###########################################################################################################################
####################################################### PARAMETERS ########################################################
###########################################################################################################################
###########################################################################################################################

# Physical Constants
RT = 8.314*298 #ideal gas law times temperature
F = 96485.0 #Faraday constant, C/mol
cmax = 7.5e-9 #mol*cm-2*s-1

# Model Parameters
k_V = 1
k_T = k_V * 1000
k_H = k_V * 1000
partialPH2 = 1
beta = [0.443716, 0.5]
conversion_factor = 1.60218e-19  # eV to J
Avo = 6.02e23  # 1/mol
GHad = -0.04 * Avo * conversion_factor
V_App = -0.3

# potential sweep & time 
UpperV = 0.05
LowerV = -0.3
scanrate = 0.025  #scan rate in V/s
timescan = (UpperV-LowerV)/(scanrate)
max_time = 60
t = np.linspace(0, 200, 1000)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]

#Initial conditions
thetaA_H0 = 0.50  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################

# Ask user for mechanism choice
while True:
    mechanism_choice = input("Choose mechanism: Volmer-Tafel (0) or Volmer-Heyrovsky (1)? ")
    if mechanism_choice in ["0", "1"]:
        break  # Exit the loop if input is valid
    print("Invalid choice. Please enter 0 or 1.")

# Convert to integer for logic checks
mechanism_choice = int(mechanism_choice)

#Linear sweep voltammetry- defining a potential as a function of time
def potential(x): return V_App
# =============================================================================
#     if x%(2*timescan)<timescan:
#             return LowerV + scanrate*(x% timescan)
#     else:
#             return UpperV - scanrate*((x - timescan) % timescan)
# =============================================================================


#Function to calculate U and Keq from theta, dG
def eqpot(theta):
        theta = np.asarray(theta)
        thetaA_star, thetaA_H = theta
        U_V = (-GHad / F) + (RT * np.log(thetaA_star / thetaA_H)) / F
        U_11 = GHad / F
        U_12 = (RT / F) * np.log(thetaA_H / thetaA_star)
        U_H = U_11 + U_12
        return U_V, U_H
    

#reduction is FORWARD, oxidation is REVERSE, all variables are consistent with this
def rates_r0(t, theta):
    theta = np.asarray(theta)
    thetaA_star, thetaA_H = theta #surface coverages again, acting as concentrations
    V = potential(t)  # Use t directly (scalar)
    U_V, U_H = eqpot(theta) #call function to find U for given theta
    
    ##Volmer Rate Equation
    r_V = k_V * (thetaA_star ** (1 - beta[0])) * (thetaA_H ** beta[0]) \
        * np.exp(beta[0] * GHad / RT) * (
        np.exp(-(beta[0]) * F * (V - U_V) / RT) - np.exp((1 - beta[0]) * F * (V - U_V) / RT))
    ##Tafel Rate Equation
    ##Tafel does not contribute to kinetic current, but does affect coverage of adsorbed hydrogen and free sites
    r_T = 0
    if mechanism_choice == 0:
        r_T = k_T * ((thetaA_H **2) - (partialPH2 * (thetaA_star ** 2) * np.exp((-2*GHad) / RT)))
    
    # Heyrovsky Rate Equation
    r_H = 0
    if mechanism_choice == 1:  
        j1 = k_H * np.exp(-beta[1] * GHad / RT) \
            * thetaA_star ** beta[1] * thetaA_H ** (1 - beta[1])
        exp21 = np.exp(-beta[1] * F * (V - U_H) / RT)
        exp22 = np.exp((1 - beta[1]) * F * (V - U_H) / RT)
        r_H = j1 * (exp21 - exp22)
    return r_V, r_T, r_H

def sitebal_r0(t, theta):
       r_V, r_T, r_H = rates_r0(t, theta)
       if mechanism_choice == 0:
            thetaStar_rate_VT = (-r_V + (2 * r_T)) / cmax
            thetaH_rate_VT = (r_V - (2 * r_T)) / cmax
            dthetadt = [(thetaStar_rate_VT), thetaH_rate_VT] # [0 = star, 1 = H]
       if mechanism_choice == 1:
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





############################################################################################################################################################
############################################################################################################################################################
########################################################## SOLVER ##########################################################################################
############################################################################################################################################################
############################################################################################################################################################
if mechanism_choice == 0:
    soln = solve_ivp(sitebal_r0, duration, theta0, t_eval= t, method = 'BDF', max_step = 0.1)
if mechanism_choice == 1:
    soln = solve_ivp(sitebal_r0, duration, theta0, t_eval = t, method = "BDF", max_step = 0.1)

print(f"Solver success: {soln.success}")
if not soln.success:
    print(soln.message)
    exit()

############################################################################################################################################################
############################################################################################################################################################
########################################################## Value Extracting ################################################################################
############################################################################################################################################################
############################################################################################################################################################

# Extract coverages from odeint
thetaA_Star = soln.y[0, :]
thetaA_H = soln.y[1, :]

avg_thetaH = np.average(thetaA_H)
print('Average Coverage:', {avg_thetaH})

#calculates rate based on theta values calculated during odeint, zips it with time given from potential(x) function
r0_vals = np.array([rates_r0(time, theta) for time, theta in zip(t, soln.y.T)])
###takes only volmer rate to compute kinetic current density
curr1 = r0_vals[:, 0] * -F

volmer_rate = r0_vals[:, 0]
tafel_rate = r0_vals[:, 1]
rH_vals = r0_vals[:, 2]

avg_rV = np.average(volmer_rate[5:])
avg_rH = np.average(rH_vals[5:])
avg_rT =  np.average(tafel_rate[5:])


# # Find the indices of the maximum and minimum values for rate
# max_curr_index = np.argmax(curr1)
# min_curr_index = np.argmin(curr1)

# # Find the corresponding times
# time_max_curr = t[max_curr_index]
# time_min_curr = t[min_curr_index]

# # Calculate the voltages at these times
# voltage_max_curr = potential(time_max_curr)
# voltage_min_curr = potential(time_min_curr)

# # Print the results
# print(f"Voltage at max current: {voltage_max_curr}")
# print(f"Voltage at min current: {voltage_min_curr}")

###########################################################################################################################
###########################################################################################################################
########################################################## PLOTS ############################################################
###########################################################################################################################
###########################################################################################################################
#Plot results
plt.figure(figsize=(8, 6))
plt.plot(V[1:], thetaA_Star[1:], label=r'$\theta_A^*$ (empty sites)', color='magenta')
plt.plot(V[1:], thetaA_H[1:], label=r'$\theta_A^H$ (adsorbed hydrogen)', color='blue')
plt.xlabel('Voltage vs. RHE (V)')
plt.ylabel('Coverage')
plt.grid()
plt.legend()
plt.title('Surface Coverage vs. Time')
plt.show()


# #Plot of reaction rate vs time
# plt.figure(figsize=(8, 6))
# plt.plot(t[1:], t_rate[1:], label='Total Rate', color='red')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$r_0$ (mol/cm²/s)')
# plt.legend()
# plt.title('Reaction Rate vs. Time')
# plt.grid()
# plt.show()

# plot kinetic current desnity as a function of potential
plt.plot(V[10:20000], curr1[10:20000], 'b')
plt.xlabel('V vs RHE(V)')
plt.ylabel('Kinetic current (mA/cm2)')
plt.title('Kinetic Current vs Potential')
plt.grid()
plt.show()

plt.plot(t[5:], volmer_rate[5:], label = f'Average Volmer Rate = {avg_rV:.4e}')
plt.xlabel('Time (s)')
plt.ylabel('rV (mol/cm^2 s)')
plt.legend()
plt.title('rV vs Time')
plt.grid()
plt.show()
    
if mechanism_choice == 0:
    plt.plot(t[5:], tafel_rate[5:], label = f'Average Tafel Rate = {avg_rT:.4e}')
    plt.xlabel('Time (s)')
    plt.ylabel('rT (mol/cm^2 s)')
    plt.legend()
    plt.title('rT vs Time')
    plt.grid()
    plt.show()

if mechanism_choice == 1:
    plt.plot(t[5:], rH_vals[5:], label = f'Average Heyrovsky Rate = {avg_rH:.4e}')
    plt.xlabel('Time (s)')
    plt.ylabel('rH (mol/cm^2 s)')
    plt.legend()
    plt.title('rH vs Time')
    plt.grid()
    plt.show()

#Create a dictionary to hold the data for excel file 
data = {
    "Time (s)": t[:len(t)],
    "Voltage (V)": V[:len(t)],  
    "Volmer Rate": r0_vals[:len(t), 0],
    "Tafel Rate": r0_vals[:len(t), 1],
    "ThetaA_Star": thetaA_Star[:len(t)],
    "ThetaA_H": thetaA_H[:len(t)],
    "Current": curr1[:len(t)]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
df.to_excel("reaction_data.xlsx", index=False)

print("Data exported successfully to reaction_data.xlsx")


