import pytest

import numpy as np

from motormetrics.ee import *

def test__get_ramp():
    arr = np.array([0, 0, 50, 50, 25, 25, -25, -25, 0, 0])
    ramps = get_ramp(arr)

    print (ramps)
    assert len(ramps) == 2