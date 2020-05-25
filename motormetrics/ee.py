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

def steady_state_error(reference, simulated, time):
    #error between reference and simulated when simulated has stablised after overshoot
    #consider last N points, they should be similar within a range and if they are take their average.
    #average - reference is steady state error
    sse_err = reference[-1] - simulated[-1]
    return sse_err, time[-1]

def overshoot(reference, simulated, minn, maxx, time):
    #value of simulated at ramp overshoot
    #negative if undershoot, positive if overshoot
    overshoot_idx = np.argmax(abs(simulated))
    overshoot_perc = 100 * (simulated[overshoot_idx] - reference[overshoot_idx]) / (maxx - minn)
    overshoot_time = time[overshoot_idx]
    return overshoot_perc, overshoot_time

def max_torque_acceleration(simulated, time):
    #maximum value of torque when speed ramp occurs
    return np.max(abs(simulated)), time[np.argmax(abs(simulated))]

def speed_drop(reference, simulated, time):
    #minimum value of speed when torque ramp occurs
    return np.max(abs(simulated - reference)), time[np.argmax(abs(simulated -reference))]

def setting_time(reference, simulated):
    #When simulated speed value is back to 0.005 of reference speed value
    pass

def speed_drop_area(reference, simulated):
    #area of speed when drop occurs
    pass

def compute_torque_metrics(experiment):
    ref_speed = experiment.reference_speed
    ref_torque = experiment.reference_torque
    ref_speed_t = experiment.speed_time
    ref_torque_t = experiment.torque_time

    ref_speed_interp = experiment.reference_speed_interp
    ref_torque_interp = experiment.reference_torque_interp

    sim_speed = experiment.speed
    sim_torque = experiment.torque

    sim_time = experiment.time

    ramp_scopes = get_ramps_from_raw_reference(ref_torque, ref_torque_t)

    ramp_start_times = []
    perc2_times = []
    perc95_times = []
    following_errs = []
    following_times = []
    overshoot_errs = []
    overshoot_times = []
    sse_errs = []
    sse_times = []
    max_trq_accs = []
    max_trq_acc_times = []

    print (len(ramp_scopes))
    for ramp_scope in ramp_scopes:
        sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scope)

        first_value = ref_speed_interp[sim_ramp_scope[0]]

        ref_speed_scope = ref_speed_interp[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        sim_speed_scope = sim_speed[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        ramp_start_times.append(sim_time[sim_ramp_scope[1]])

        ref_speed_scope, sim_speed_scope = mirror(ref_speed_scope, sim_speed_scope, first_value)

        perc2_time = response_time_2perc(ref_speed_scope,
                            sim_speed_scope, sim_time_scope)
        perc2_times.append(round(perc2_time, 5))

        perc95_time = response_time_95perc(ref_speed_scope,
                            sim_speed_scope, sim_time_scope)
        perc95_times.append(round(perc95_time, 5))

        following_err, following_time = following_error(ref_speed_scope,
                                        sim_speed_scope, sim_time_scope)
        following_errs.append(round(following_err,4))
        following_times.append(round(following_time, 5))

        minn = min(ref_speed_scope)
        maxx = max(ref_speed_scope)

        ref_speed_scope = ref_speed_interp[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
        sim_speed_scope = sim_speed[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]

        ref_speed_scope, sim_speed_scope = mirror(ref_speed_scope, sim_speed_scope,
                                                    first_value)

        overshoot_err, overshoot_time = overshoot(ref_speed_scope, sim_speed_scope,
                                        minn, maxx, sim_time_scope)

        overshoot_errs.append(round(overshoot_err,4))
        overshoot_times.append(round(overshoot_time, 5))

        sse_err, sse_time = steady_state_error(ref_speed_scope, sim_speed_scope,
                                                sim_time_scope)

        sse_errs.append(round(sse_err, 4))
        sse_times.append(round(sse_time, 5))

        sim_torque_scope = sim_torque[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

        max_trq_acc, max_trq_acc_time = max_torque_acceleration(sim_torque_scope,
                                        sim_time_scope)

        max_trq_accs.append(round(max_trq_acc, 4))
        max_trq_acc_times.append(round(max_trq_acc_time, 5))

    return {'perc2_times': perc2_times,
            'perc95_times': perc95_times,
            'following_errs': following_errs,
            'following_times': following_times,
            'overshoot_errs': overshoot_errs,
            'overshoot_times': overshoot_times,
            'ramp_start_times': ramp_start_times,
            'sse_errs': sse_errs,
            'sse_times': sse_times,
            'max_trq_accs': max_trq_accs,
            'max_trq_acc_times': max_trq_acc_times}

def compute_speed_metrics(experiment):
    ref_speed = experiment.reference_speed
    ref_torque = experiment.reference_torque
    ref_speed_t = experiment.speed_time
    ref_torque_t = experiment.torque_time

    ref_speed_interp = experiment.reference_speed_interp
    ref_torque_interp = experiment.reference_torque_interp

    sim_speed = experiment.speed
    sim_torque = experiment.torque

    sim_time = experiment.time

    ramp_scopes = get_ramps_from_raw_reference(ref_speed, ref_speed_t)

    ramp_start_times = []
    perc2_times = []
    perc95_times = []
    following_errs = []
    following_times = []
    overshoot_errs = []
    overshoot_times = []
    sse_errs = []
    sse_times = []
    speed_drops = []
    speed_drops_times = []

    print (len(ramp_scopes))
    for ramp_scope in ramp_scopes:
        sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scope)

        first_value = ref_torque_interp[sim_ramp_scope[0]]

        ref_torque_scope = ref_torque_interp[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        sim_torque_scope = sim_torque[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[1]: sim_ramp_scope[-1] + 1]
        ramp_start_times.append(sim_time[sim_ramp_scope[1]])

        ref_torque_scope, sim_torque_scope = mirror(ref_torque_scope, sim_torque_scope, first_value)

        perc2_time = response_time_2perc(ref_torque_scope,
                            sim_torque_scope, sim_time_scope)
        perc2_times.append(round(perc2_time, 5))

        perc95_time = response_time_95perc(ref_torque_scope,
                            sim_torque_scope, sim_time_scope)
        perc95_times.append(round(perc95_time, 5))

        following_err, following_time = following_error(ref_torque_scope,
                                        sim_torque_scope, sim_time_scope)
        following_errs.append(round(following_err,4))
        following_times.append(round(following_time, 5))

        minn = min(ref_torque_scope)
        maxx = max(ref_torque_scope)

        ref_torque_scope = ref_torque_interp[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
        sim_torque_scope = sim_torque[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]

        ref_torque_scope, sim_torque_scope = mirror(ref_torque_scope, sim_torque_scope,
                                                    first_value)

        overshoot_err, overshoot_time = overshoot(ref_torque_scope, sim_torque_scope,
                                        minn, maxx, sim_time_scope)

        overshoot_errs.append(round(overshoot_err,4))
        overshoot_times.append(round(overshoot_time, 5))

        sse_err, sse_time = steady_state_error(ref_torque_scope, sim_torque_scope,
                                                sim_time_scope)

        sse_errs.append(round(sse_err, 4))
        sse_times.append(round(sse_time, 5))

        sim_speed_scope = sim_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
        sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

        spd_drp, spd_drp_time = speed_drop(sim_speed_scope,
                                        sim_time_scope)

        speed_drops.append(round(spd_drp, 4))
        speed_drops_times.append(round(spd_drp_time, 5))

    return {'perc2_times': perc2_times,
            'perc95_times': perc95_times,
            'following_errs': following_errs,
            'following_times': following_times,
            'overshoot_errs': overshoot_errs,
            'overshoot_times': overshoot_times,
            'ramp_start_times': ramp_start_times,
            'sse_errs': sse_errs,
            'sse_times': sse_times,
            'speed_drops': speed_drops,
            'speed_drops_times': speed_drops_times}
