# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test piecewise constant function
"""

import pytest
import mindspore.context as context
import mindspore.numpy as mnp

from mind_quant_finance.math import piecewise 

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

@pytest.mark.level0
def test_piecewise_constant_value_no_batch():
    
    for dtype in [mnp.float32]:
        x = mnp.array([0., 0.1, 2. ,11.])
        jump_locations = mnp.array([0.1, 10], dtype=dtype)
        values = mnp.array([3, 4, 5], dtype=dtype)
        piecewise_func = piecewise.PiecewiseConstantFunction(jump_locations, values, dtype=dtype)
        computed_value = piecewise_func(x)
        expected_value = mnp.array([3., 3., 4., 5.])
        
        isclose = mnp.isclose(computed_value, expected_value, 1e-6)
        assert isclose == [True, True, True, True]