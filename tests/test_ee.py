import pytest

import numpy as np
import scipy.io as sio

from motorrefgen.config import ExperimentConfig
from motorrefgen.experiment import Experiment

from motorsim.simconfig import SimConfig
from motorsim.simulators.conn_python import Py2Mat

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

    assert abs(perc95_time - 1.45) <= 0.000001

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

    following_err, following_time = following_error(ref_speed_scope, sim_speed_scope, sim_time_scope)

    assert abs(following_err - 2.46235934) <= 0.000001
    assert abs(following_time - 1.25) <= 0.000000001


def test__overshoot():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    minn = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1].min()
    maxx = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1].max()
    overshoot_err, overshoot_time = overshoot(ref_speed_scope, sim_speed_scope,
                                    minn, maxx, sim_time_scope)

    assert abs(overshoot_err - 4.40504014) <= 0.000001
    assert abs(overshoot_time - 1.6) <= 0.000000001

def test__stead_state_error():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    ref_speed_scope = ref_speed[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    sim_speed_scope = sim_speed[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[2]: sim_ramp_scope[-1] + 1]
    minn = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1].min()
    maxx = ref_speed[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1].max()
    sse_err, sse_time = steady_state_error(ref_speed_scope, sim_speed_scope,
                                            minn, maxx, sim_time_scope)

    assert abs(sse_err - 0.57655071) <= 0.000001
    assert abs(sse_time - 2) <= 0.000000001

def test__max_torque_acceleration():
    data = sio.loadmat('tests/test1.mat')

    ref_speed_inp = data['RefSpeedInp'][0]
    ref_speed_inp_t = data['RefSpeedInp_t'][0]

    ref_speed = data['RefSpeed']
    sim_speed = data['Speed']
    sim_torque = data['Torque']

    sim_time = data['SimTime']

    ramp_scopes = get_ramps_from_raw_reference(ref_speed_inp, ref_speed_inp_t)
    sim_ramp_scope = get_ramp_from_sim_reference(sim_time, ramp_scopes[0])

    sim_torque_scope = sim_torque[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]
    sim_time_scope = sim_time[sim_ramp_scope[0]: sim_ramp_scope[-1] + 1]

    max_trq_acc, max_trq_acc_time = max_torque_acceleration(sim_torque_scope, sim_time_scope)

    assert abs(max_trq_acc - 16.343483203104014) <= 0.000001
    assert abs(max_trq_acc_time - 1.1) <= 0.000000001

def test__compute_metrics():
    config = ExperimentConfig()
    experiment = Experiment(config=config)
    simconfig = SimConfig()
    simconfig.set_config_from_json({'Data_Ts': 0.001})
    simulator = Py2Mat(simconfig)

    reference = {'reference_speed': [0,0,50,50],
                 'speed_time': [0,1,2,3],
                 'reference_torque': [0,0,0,0],
                 'torque_time': [0,1,2,3]}

    experiment.set_manual_reference(reference)
    experiment.simulate(simulator)

    metrics = compute_metrics(experiment)

    assert 'perc2_times' in metrics
    assert 'perc95_times' in metrics
    assert 'following_errs' in metrics
    assert 'following_times' in metrics
    assert 'overshoot_errs' in metrics
    assert 'overshoot_times' in metrics
    assert 'ramp_start_times' in metrics
    assert 'sse_errs' in metrics
    assert 'sse_times' in metrics
    assert 'max_trq_accs' in metrics
    assert 'max_trq_acc_times' in metrics

    assert len(metrics['perc2_times']) == 1
    assert metrics['perc2_times'][0] == 0.045
    assert metrics['perc95_times'][0] == 0.96
