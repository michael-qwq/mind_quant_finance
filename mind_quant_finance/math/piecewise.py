import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mindspore import ms_function


@ms_function
def _piecewise_constant_function(x, jump_locations, values, side='left'):

    indices = mnp.searchsorted(jump_locations, x, side=side)
    axis = 0
    res = P.Gather()(values, indices, axis)

    return res


class PiecewiseConstantFunction(nn.Cell):

    def __init__(self, jump_locations, values, dtype=None):

        super(PiecewiseConstantFunction, self).__init__()

        self._jump_locations = mnp.array(jump_locations, dtype=dtype)
        self._dtype = dtype or self._jump_locations
        self._values = mnp.array(values, dtype=self._dtype)
        self._piecewise_constant_function = _piecewise_constant_function

    def construct(self, x, left_continuous=True):
        return self._piecewise_constant_function(x, self._jump_locations, self._values)
