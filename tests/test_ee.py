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
