import numpy as np
from tabulate import tabulate

# # potential sweep & time
UpperV = 0
LowerV = -0.2
scanrate = 0.05  #scan rate in V/s
timescan = 2*(UpperV-LowerV)/(scanrate)
max_time = timescan
t = np.arange(0.0, max_time, scanrate)
endtime = t[-1]
duration = [0, endtime]
time_index = [t]

#Initial conditions
thetaA_H0 = 0.99  # Initial coverage of Hads, needs to be high as this is reduction forward
thetaA_Star0 = 1.0 - thetaA_H0  # Initial coverage of empty sites
theta0 = np.array([thetaA_Star0, thetaA_H0])

forward1 = []
backward1 = []
forward2 = []
backward2 = []

############################################################################################################################
############################################################################################################################
########################################################## FUNCTIONS #######################################################
############################################################################################################################
############################################################################################################################

#Linear sweep voltammetry- defining a potential as a function of time
def potential1(x):
    #timescan above is the same as single_sweep_time
    single_sweep_time = (UpperV - LowerV) / scanrate
    cycle_time = 2 * single_sweep_time

    t_in_cycle = x % cycle_time

    if t_in_cycle < single_sweep_time: #forward
        forward = UpperV - scanrate * t_in_cycle
        forward1.append(forward)
        backward1.append(None)
        return forward
    else: #reverse
        backward = LowerV + scanrate * (t_in_cycle - single_sweep_time)
        backward1.append(backward)
        forward1.append(None)
        return backward

def potential2(x):
    if x%(2*timescan)<timescan:
        ValF = UpperV - scanrate*((x - timescan) % timescan)
        forward2.append(ValF)
        backward2.append(None)
        return ValF
    else:
        ValB = LowerV + scanrate*(x% timescan)
        backward2.append(ValB)
        backward2.append(None)
        return ValB

for ti in t:
    potential1(ti)
    potential2(ti)

# Tabulate output
table_data = list(zip(t, forward1, backward1, forward2, backward2))
headers = ["Time", "Potential 1 Forward", "Potential 1 Backward", "Potential 2 Forward", "Potential 2 Backward"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))