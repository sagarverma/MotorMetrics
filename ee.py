import math

import numpy as np
import scipy.io as sio

def mirror(reference, simulated):
    if simulated.min() < 0:
        return abs(reference), abs(simulated)
    return reference, simualted

def get_ramp(simulated):
    ramp_start = np.argmin(simulated == simulated.min())
    ramp_end = np.argmax(simulated)
    return ramp_start, ramp_end

def response_time_2perc(reference, simulated, time):
    #when is the simulated quantity 2% of the nominal reference quantity.
    start, end = get_ramp(simulated)
    perc2_time = time[start + np.argmax(reference[start:end] >=\
                        0.02 * simulated.max())] - time[start]
    return perc2_time[0]

def response_time_95perc(reference, simulated, time):
    #when is the simulated quantity 95% of the nominal reference quantity
    start, end = get_ramp(simulated)
    perc2_time = time[start + np.argmax(reference[start:end] >= 0.95 * simulated.max())] - time[start]
    return perc2_time[0]

def following_error(reference, simulated):
    #error between refernece and simulated when reference is 0.5 of of the nominal
    start, end = get_ramp(simulated)
    following_indx = start + np.argmax(simulated >= 0.5 * (simulated.max()-simulated.min()))
    following_err = reference[following_indx] - simulated[following_indx]
    return following_err[0]

def stead_state_error(reference, simulated):
    #error between reference and simulated when simulated has stablised after overshoot
    #consider last N points, they should be similar within a range and if they are take their average. 
    #average - reference is steady state error
    pass

def overshoot(reference, simulated):
    #value of simulated at ramp overshoot
    start, end = get_ramp(simulated)
    overshoot_idx = end + np.argmax(abs(reference[end:-1] - simulated[end:-1]))
    overshoot_perc = 100 * (simulated[overshoot_idx] - reference[overshoot_idx]) / (simulated.max() - simulated.min())
    return overshoot_perc[0]

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


test = sio.loadmat('../mat_sim/test1.mat')
print (test.keys())
sim_speed = test['Speed']
ref_speed = test['RefSpeed']
sim_torque = test['Torque']
ref_torque = test['RefLoad']
time = test['t']

#mirror is not the correct solution
# ref_speed, sim_speed = mirror(ref_speed, sim_speed)
# ref_torque, sim_torque = mirror(ref_torque, sim_torque)

print ('2 % response time', response_time_2perc(ref_speed, sim_speed, time), 'seconds')
print ('95 % response time', response_time_95perc(ref_speed, sim_speed, time), 'seconds')
print ('following error', following_error(ref_speed, sim_speed))
print ('overshoot', overshoot(ref_speed, sim_speed))
print ('max torque acceleration', max_torque_acceleration(sim_torque))
