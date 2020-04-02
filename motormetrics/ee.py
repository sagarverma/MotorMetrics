import math

import numpy as np
import scipy.io as sio

def mirror(reference, simulated, first_value):
    reference = abs(reference - first_value)
    simulated = abs(simulated - first_value)
    return reference, simulated

def get_ramps_from_raw_reference(reference_data, reference_time):
    #Use reference without interpolation and also take in data ts
    #Find all change points convert it back to the array location in
    #simulation using the reference time
    ramp_scopes = []
    for t in range(1, len(reference_data)-1):
        if reference_data[t] != reference_data[t+1]:
            ramp_scopes.append(reference_time[t-1: t+3])
    return ramp_scopes

def get_ramp_from_sim_reference(sim_time, ramp_scope):
    sim_ramp_scope = [np.where(sim_time == x)[0][0] for x in ramp_scope]
    return sim_ramp_scope

def response_time_2perc(reference, simulated, time):
    #when is the simulated quantity 2% of the nominal reference quantity.
    perc2_time = time[np.argmax(simulated >= 0.02 * reference.max())] - time[0]
    return perc2_time

def response_time_95perc(reference, simulated, time):
    #when is the simulated quantity 95% of the nominal reference quantity
    if np.where(simulated >= 1.05 * reference.max()):
        inside_bools = (simulated >= 0.95 * reference.max()) &\
                    (simulated <= 1.05 * reference.max())
        inside_bools = inside_bools.flatten().tolist()
        inside_bools.reverse()
        stable_index = len(inside_bools) - np.argmin(inside_bools) - 1
        perc95_time = time[stable_index] - time[0]
    else:
        perc95_time = time[np.argmax(simulated >= 0.95 * reference.max())] - time[0]

    return perc95_time

def following_error(reference, simulated, time):
    #error between refernece and simulated when reference is 0.5 of of the nominal
    following_indx = np.argmax(reference >= 0.5 * (reference.max()-reference.min()))
    following_err = reference[following_indx] - simulated[following_indx]
    following_time = time[following_indx]
    return following_err, following_time

def stead_state_error(reference, simulated):
    #error between reference and simulated when simulated has stablised after overshoot
    #consider last N points, they should be similar within a range and if they are take their average.
    #average - reference is steady state error
    pass

def overshoot(reference, simulated, minn, maxx, time):
    #value of simulated at ramp overshoot
    #negative if undershoot, positive if overshoot
    overshoot_idx = np.argmax(abs(reference - simulated))
    overshoot_perc = 100 * (reference[overshoot_idx] - simulated[overshoot_idx]) / (minn - maxx)
    overshoot_time = time[overshoot_idx]
    return overshoot_perc, overshoot_time

def max_torque_acceleration(simulated):
    #maximum value of torque when speed ramp occurs
    return np.max(simulated)

def speed_drop(reference, simulated):
    #minimum value of speed when torque ramp occurs
    pass

def setting_time(reference, simulated):
    #When simulated speed value is back to 0.005 of reference speed value
    pass

def speed_drop_area(reference, simulated):
    #area of speed when drop occurs
    pass
