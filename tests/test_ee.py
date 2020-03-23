import pytest

import numpy as np

from motormetrics.ee import *

def test__get_ramps():
    speed = np.array([0, 0, 50, 50, 25, 25, -25, -25, 0, 0])
    time = np.array([0, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6])
    ramps = get_ramps(speed, time)

    assert len(ramps) == 4
    assert ramps[0] == [0, 2]
    assert ramps[1] == [1.5, 3]
    assert ramps[2] == [2.5, 4.5]
    assert ramps[3] == [4, 6]
