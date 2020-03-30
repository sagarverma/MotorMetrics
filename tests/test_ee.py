import pytest

import numpy as np
import scipy.io as sio

from motormetrics.ee import *

def test__get_ramps_from_raw_reference():
    speed = np.array([0, 0, 50, 50, 25, 25, -25, -25, 0, 0])
    time = np.array([0, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6])
    ramps = get_ramps_from_raw_reference(speed, time)

    assert len(ramps) == 4
    assert np.array_equal(ramps[0], [0, 1, 1.5, 2])
    assert np.array_equal(ramps[1], [1.5, 2, 2.5, 3])
    assert np.array_equal(ramps[2], [2.5, 3, 4, 4.5])
    assert np.array_equal(ramps[3], [4, 4.5, 5, 6])

def test_get_ramp_from_sim_reference():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    assert len(sim_ramp_scope) == 4
    assert sim_time[sim_ramp_scope[0]] == ramp_scopes[0][0]
    assert sim_time[sim_ramp_scope[1]] == ramp_scopes[0][1]
    assert sim_time[sim_ramp_scope[2]] == ramp_scopes[0][2]
    assert sim_time[sim_ramp_scope[3]] == ramp_scopes[0][3]

def test__response_time_2perc():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

    perc2_time = response_time_2perc(ref_speed_scope, sim_speed_scope, sim_time_scope)

    assert abs(perc2_time - 1.05) <= 0.000001

def test__response_time_95perc():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

    perc95_time = response_time_95perc(ref_speed_scope, sim_speed_scope, sim_time_scope)

    assert abs(perc95_time - 1.5) <= 0.000001

def test__following_error():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

    following_err, following_time = following_error(ref_speed_scope, sim_speed_scope, sim_time)

    assert abs(following_err - 1.25352951) <= 0.000001
    assert abs(following_time - 1.3) <= 0.000000001


def test__overshoot():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

    overshoot_err, overshoot_time = overshoot(ref_speed_scope, sim_speed_scope, sim_time)

    assert abs(overshoot_err - -4.71584292) <= 0.000001
    assert abs(overshoot_time - 1.05) <= 0.000000001
