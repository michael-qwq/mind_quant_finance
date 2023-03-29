# Copyright 2022 Huawei Technologies Co., Ltd
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
Note:
    This is interface that is subject to change or future or add multiple related European option pricing method.
"""
import mindspore
import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops as P
from mindspore import Tensor

import random
from typing import Optional, Any


Dtype = Any


class MonteCarloSimulationPricing(nn.Cell):
    """
        European option Monte Carlo method based on Brownian motion model.
        Returns two or three Tensors, "option price","option value" or add an "option price change Process".

        .. warning::
            Warning description.

        Note:
            Note description.

        .. math::

            d_St = (r-q)*St*dt + σ*St*dzt → △St = St*(r-q)*△t + σ*St*ε*sqrt(△t).

            Where "r" is the continuous compound risk-free interest rate.
            by the price limit,
            take the logarithm on both sides of the above formula.

        Args:
            dim(int): Dimension; asset quantity.
            batchsize(int): the size of a simulation.
            The macro environment of each simulation is constant.
            (you can understand that the same economic macro environment: growth, recession, depression and prosperity).
            Currently, relevant interfaces are not enabled.
            num_ paths(int): number of simulations in each batchsize.
            Process(Bool): whether to output the price change process, which can be used for data export
                and visualization.

        Returns:
            Type, description.

        Raises:
            Type: Description.

        Inputs:
            - **pv** (Union[Tensor, List[float]]): - present value of assets, shape is (dim, 1).
              For example, [2.1, 1, 8.7] represents three assets, which are
              the current prices of 2.1, 1 and 8.7 respectively.
            - **sigma** (Union[Tensor, List[float]]): - σ,asset volatility.
                The shape is the same as pv (dim,1) , corresponding to the volatility of various assets of pv.
            - **q** (Union[Tensor, List[float]]): - Return on assets
                The shape is the same as pv (dim,1), which corresponds to the yield of each asset of pv.
                If the asset has no dividend, it is [0, 0, 0.]
            - **rf** (Union[Tensor, List[float]]): - risk-free rate of return.
                If the risk-free rate of return is fixed, the shape is (1,1).
                If it fluctuates with time, the incoming shape is (num_ timesteps,1).
            - **num_ timesteps** (int): the number of simulation executions within the specified time.
            - **dt(float)** : time step, interval size of each simulation time. Default: "1.".
            - **lr(float)** : price rate limit. Default: "0.".

        Outputs:
            - **Option value** (Union[Tensor, List[float]]): - The shape is (batchsize, num_paths, dim)
            - **Option price** (Union[Tensor, List[float]]): - The shape is (batchsize, num_paths, dim)
            ** Change value of option value** (Union[Tensor, List[float]]): -:value record matrix of option
            (if "Process" = "True"). after change in each "dt", (batchsize, num_paths, num_timesteps ,dim)

        Examples:
            Please refer to the notebook file of example

        Supported Platforms:
            ``GPU``

    """

    def __init__(self,
                 dim: int,
                 batchsize: int,
                 num_paths: int,
                 Process=False,
                 seed: int = 0,
                 dtype: Dtype = mstype.float32):
        super(MonteCarloSimulationPricing, self).__init__()
        self.std = P.StandardNormal(seed, seed)
        self.sqrt = P.Sqrt()
        self.square = P.Square()
        self.einsum = P.Einsum("abcd->abd")
        self.batchsize = batchsize
        self.num_paths = num_paths
        self.dim = dim
        self.process = Process
        self.scalar_to_tensor = P.ScalarToTensor()
        self.dtype = dtype

    def construct(self, pv, sigma, q, rf, num_timesteps, dt, lr=0, times=None):
        '''
            Core method for calculating European options.
        '''
        if len(pv) != self.dim:
            print('The asset quantity does not match the given present value quantity')
            return None
        else:
            pv = Tensor(pv, self.dtype)
            sigma = Tensor(sigma, self.dtype)
            q = Tensor(q, self.dtype)
            rf = Tensor(rf, self.dtype)
            pv = np.broadcast_to(
                pv, (self.batchsize, self.num_paths, self.dim))
            sigmas = np.broadcast_to(
                sigma, (self.batchsize, self.num_paths, num_timesteps, self.dim))
            square_sigmas = self.square(sigmas)
            rfs = np.broadcast_to(
                rf, (self.batchsize, self.num_paths, num_timesteps, self.dim))
            qs = np.broadcast_to(
                q, (self.batchsize, self.num_paths, num_timesteps, self.dim))
            dts = np.broadcast_to(Tensor(
                dt, self.dtype), (self.batchsize, self.num_paths, num_timesteps, self.dim))
            sqrt_dts = self.sqrt(dts)
            eps = self.std(
                (self.batchsize, self.num_paths, num_timesteps, self.dim))
            ln_S = (rfs-qs-(square_sigmas)/2)*dts+sigmas*eps*sqrt_dts
            ln_SC = np.copy(ln_S)
            lr = self.scalar_to_tensor(lr, mindspore.float32)
            if lr != 0:
                ln_S = np.where(P.Abs()(ln_SC) > lr, lr, ln_S)
            if self.process is True:
                num_timesteps = Tensor(num_timesteps, mindspore.int32)
                output_process = P.ZerosLike()(ln_S)
                i = self.scalar_to_tensor(0, mindspore.int32)
                while i < num_timesteps:
                    dln_S = ln_S[:, :, i, :]
                    if i == 0:
                        output_process[:, :, i, :] = pv*(dln_S+1)
                    else:
                        output_process[:, :, i,
                                       :] = output_process[:, :, i-1, :]*(dln_S+1)
                    i += 1
                return P.Mul()(self.einsum([ln_S])+1, pv), P.Mul()(self.einsum([ln_S])+1, pv)-pv, output_process
            else:
                return P.Mul()(self.einsum([ln_S])+1, pv), P.Mul()(self.einsum([ln_S])+1, pv)-pv
