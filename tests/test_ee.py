import pytest

import numpy as np

from motormetrics.ee import *

def test__get_ramp():
    arr = np.array([0, 0, 50, 50])
    ramp_start, ramp_end = get_ramp(arr)
    
    assert ramp_start == 1
    assert ramp_end == 2