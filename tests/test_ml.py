import pytest

import numpy as np

from motormetrics.ml import *

def test__flatten_extra_dims():
    arr = np.zeros((100,3))
    assert flatten_extra_dims(arr).shape == (300,)

def test__r2():
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 0, 0])

    assert r2(true, pred) == 0

    pred = np.array([[0, 0], [1, 1], [0, 1]])
    true = np.array([[0, 0], [1, 1], [0, 1]])

    assert r2(true, pred) == 1

    pred = np.array([0, 1])
    true = np.array([0, 1, 0])

    with pytest.raises(AssertionError):
        r2(true, pred)

def test__rmsle():
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 0, 0])

    assert abs(rmsle(true, pred) - 0.346574) <= 0.00001

    pred = np.array([[0, 0], [1, 1], [0, 1]])
    true = np.array([[0, 0], [1, 1], [0, 1]])

    assert rmsle(true, pred) == 0

    pred = np.array([-1, -1, -1])
    true = np.array([-2, -2, -2])

    with pytest.raises(AssertionError):
        rmsle(true, pred)

    pred = np.array([0, 1])
    true = np.array([0, 1, 0])

    with pytest.raises(AssertionError):
        rmsle(true, pred)

def test__rmse():
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 0, 0])

    assert rmse(true, pred) == 0.5

    pred = np.array([[0, 0], [1, 1], [0, 1]])
    true = np.array([[0, 0], [1, 1], [0, 1]])

    assert rmse(true, pred) == 0

    pred = np.array([-1, -1, -1])
    true = np.array([-2, -2, -2])

    assert rmse(true, pred) == 1

    pred = np.array([0, 1])
    true = np.array([0, 1, 0])

    with pytest.raises(AssertionError):
        rmse(true, pred)


def test__mae():
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 0, 0])

    assert mae(true, pred) == 0.25

    pred = np.array([[0, 0], [1, 1], [0, 1]])
    true = np.array([[0, 0], [1, 1], [0, 1]])

    assert mae(true, pred) == 0

    pred = np.array([-1, -1, -1])
    true = np.array([-2, -2, -2])

    assert mae(true, pred) == 1

    pred = np.array([0, 1])
    true = np.array([0, 1, 0])

    with pytest.raises(AssertionError):
        mae(true, pred)

def test__smape():
    pred = np.array([0, 0, 0, 1])
    true = np.array([0, 0, 0, 0])

    assert abs(smape(true, pred) - 49.9995) <= 0.0001

    pred = np.array([[0, 0], [1, 1], [0, 1]])
    true = np.array([[0, 0], [1, 1], [0, 1]])

    assert smape(true, pred) == 0

    pred = np.array([-1, -1, -1])
    true = np.array([-2, -2, -2])

    assert abs(smape(true, pred) - 66.6664) <= 0.0001

    pred = np.array([0, 1])
    true = np.array([0, 1, 0])

    with pytest.raises(AssertionError):
        smape(true, pred)

def test__sc():
    arr = np.array([1, 0, 1, 0])

    assert sc(arr) == 1

def test__smape_vs_sc():
    pred = np.array([0, 1, 0, 1, 0, 1, 1])
    true = np.array([0, 1, 0, 1, 0, 1, 1])
    window = 2

    val = smape_vs_sc(true, pred, window)
    assert val.shape == (true.shape[0]-window-1, 2)

    assert val[0][0] == val[1][0] and val[0][1] == val[1][1]

    window = 0
    with pytest.raises(AssertionError):
        smape_vs_sc(true, pred, window)
