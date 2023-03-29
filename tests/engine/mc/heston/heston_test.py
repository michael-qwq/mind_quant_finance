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
test monte carlo simulation for heston model
"""

import pytest

import numpy as np
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore import context

from mind_quant_finance.math import piecewise
from mind_quant_finance.engine.mc.heston import HestonModel


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

@pytest.mark.level0
def test_constant_parameters_heston():

    dtype = mnp.float32
    theta = 0.5
    process = HestonModel(mean_reversion=1.0, theta=theta,
                          volvol=1.0, rho=-0.0, dtype=dtype)
    years = 1.0
    times = Tensor(np.linspace(0.0, years, int(10 * years)))
    num_paths = 3

    paths = process(
        initial_state=Tensor(np.array([np.log(100), 0.45])),
        num_timesteps=len(times),
        num_paths=num_paths,
        times=times
    )

    assert paths.shape == (2, 9, 3)


@pytest.mark.level0
def test_function_parameters_heston():

    dtype = mnp.float32
    mean_reversion = piecewise.PiecewiseConstantFunction(
        jump_locations=[0.5], values=[1, 1.1], dtype=dtype)
    theta = piecewise.PiecewiseConstantFunction(
        jump_locations=[0.5], values=[1, 0.9], dtype=dtype)
    volvol = piecewise.PiecewiseConstantFunction(
        jump_locations=[0.3], values=[0.1, 0.2], dtype=dtype)
    rho = piecewise.PiecewiseConstantFunction(
        jump_locations=[0.5], values=[0.4, 0.6], dtype=dtype)
    process = HestonModel(mean_reversion=mean_reversion,
                          theta=theta, volvol=volvol, rho=rho, dtype=dtype)
    years = 1.0
    times = Tensor(np.linspace(0.0, years, int(10 * years)))
    num_paths = 3
    paths = process(
        initial_state=Tensor(np.array([np.log(100), 0.45])),
        num_timesteps=len(times),
        num_paths=num_paths,
        times=times
    )

    assert paths.shape == (2, 9, 3)

