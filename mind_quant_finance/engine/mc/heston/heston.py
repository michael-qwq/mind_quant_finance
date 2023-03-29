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
In finance, the Heston model, named after Steven L. Heston, 
is a mathematical model that describes the evolution of the volatility of an underlying asset.
It is a stochastic volatility model: such a model assumes that the volatility of the asset is not constant, nor even deterministic, but follows a random process.
"""

import mindspore.numpy as mnp
import mindspore.ops as P
import mindspore.nn as nn

from mind_quant_finance.math import piecewise

__all__ = [
    "HestonModel",
]


class HestonModel(nn.Cell):
    """
    Monte Carlo Simulation For Heston Model


    Args:
        initial_state (mindspore.Tensor): initial state of log spot and vol
        num_timesteps (int): number of time steps to simulate
        num_paths (int): number of paths to simulate
        tolerance (float): tolerance of the adjoint time gap
        times (mindspore.Tensor): time grid

    Returns:
        mindspre.Tensor, path of log spot and variance drift
    
    Examples:
    >>> from mind_quant_finance.engine.mc.heston import HestonModel
    >>> dtype = mnp.float32
    >>> seed = 1
    >>> theta = 0.5
    >>> process = HestonModel(mean_reversion=1.0, theta=theta,
                          volvol=1.0, rho=-0.0, dtype=dtype)
    >>> years = 1.0
    >>> times = Tensor(np.linspace(0.0, years, int(10 * years)))
    >>> num_paths = 3

    >>> paths = process(
    ... initial_state=Tensor(np.array([np.log(100), 0.45])),
    ... num_timesteps=len(times),
    ... num_paths=num_paths,
    ... seed=seed,
    ... times=times)
    >>> print(path)
    [[[4.8156638e+00 4.6070127e+00 3.8259201e+00]
    [4.5860720e+00 4.9500113e+00 3.7881947e+00]
    [4.7622180e+00 4.9643111e+00 3.7010713e+00]
    [4.7163038e+00 5.2245984e+00 3.2445865e+00]
    [4.7210646e+00 5.1865859e+00 3.3585775e+00]
    [4.7144003e+00 5.4142361e+00 3.3605635e+00]
    [4.5891528e+00 5.6028652e+00 3.2838130e+00]
    [4.2114220e+00 5.8802009e+00 3.0929790e+00]
    [3.8681121e+00 5.7577729e+00 3.0442882e+00]]
    """

    def __init__(self, mean_reversion, theta, volvol, rho, dtype):
        super(HestonModel, self).__init__()

        self._dtype = dtype or mnp.float32
        self._dim = 2

        if isinstance(mean_reversion, piecewise.PiecewiseConstantFunction):
            self._mean_reversion = mean_reversion
        else:
            self._mean_reversion = mnp.asarray(
                mean_reversion, dtype=self._dtype)

        if isinstance(theta, piecewise.PiecewiseConstantFunction):
            self._theta = theta
        else:
            self._theta = mnp.asarray(theta, dtype=self._dtype)

        if isinstance(volvol, piecewise.PiecewiseConstantFunction):
            self._volvol = volvol
        else:
            self._volvol = mnp.asarray(volvol, dtype=self._dtype)

        if isinstance(rho, piecewise.PiecewiseConstantFunction):
            self._rho = rho
        else:
            self._rho = mnp.asarray(rho, dtype=self._dtype)

        def _vol_fn(t, x):
            """create vol matrix"""
            t = mnp.asarray([t], dtype=self._dtype)
            vol = mnp.sqrt(mnp.abs(x[..., 1]))
            zeros = mnp.zeros_like(vol)
            rho, volvol = self._get_parameters(t, self._rho, self._volvol)
            rho, volvol = rho[0], volvol[0]
            vol_matrix_2 = mnp.stack(
                [zeros, volvol * mnp.sqrt(1 - rho ** 2) * vol], -1)
            vol_matrix_1 = mnp.stack([vol, volvol * rho * vol], -1)
            vol_matrix = mnp.stack([vol_matrix_1, vol_matrix_2])
            return vol_matrix

        def _drift_fn(t, x):
            """create drift matrix"""
            t = mnp.asarray([t], dtype=self._dtype)
            var = x[..., 1]
            mean_reversion, theta = self._get_parameters(
                t, self._mean_reversion, self._theta)

            mean_reversion, theta = mean_reversion[0], theta[0]
            log_spot_drift = -var/2
            var_drift = mean_reversion * (theta - var)
            drift = mnp.stack([log_spot_drift, var_drift], -1)
            return drift

        self._drift_fn = _drift_fn
        self._vol_fn = _vol_fn

    def construct(
        self,
        initial_state,
        num_timesteps,
        num_paths=1,
        tolerance=1e-6,
        times=None
    ):

        times = mnp.asarray(times, dtype=self._dtype)

        normal_draws = mnp.randn(
            (num_timesteps, num_paths, self._dim), dtype=self._dtype)

        current_log_spot = (
            mnp.asarray(initial_state[..., 0], dtype=self._dtype) *
            mnp.ones(shape=[num_paths], dtype=self._dtype)
        )
        current_vol = (
            mnp.ones(shape=[num_paths], dtype=self._dtype) *
            mnp.asarray(initial_state[..., 1], dtype=self._dtype)
        )

        return self._sample_path(times=times,
                                 current_log_spot=current_log_spot,
                                 current_vol=current_vol,
                                 num_paths=num_paths,
                                 tolerance=tolerance,
                                 normal_draws=normal_draws)

    def _sample_path(
        self,
        times,
        current_log_spot,
        current_vol,
        num_paths,
        tolerance,
        normal_draws
    ):

        dt = times[1:] - times[:-1]

        mean_reversion, theta, volvol, rho = self._get_parameters(
            times + P.ReduceMin()(dt)/2, self._mean_reversion, self._theta, self._volvol, self._rho)

        steps_num = dt.shape[-1]

        log_spot_paths = mnp.zeros(
            shape=[steps_num, num_paths], dtype=self._dtype)

        vol_paths = mnp.zeros(
            shape=[steps_num, num_paths], dtype=self._dtype)

        vol_paths[0] = mnp.zeros(shape=[num_paths], dtype=self._dtype)
        log_spot_paths[0] = mnp.zeros(shape=[num_paths], dtype=self._dtype)

        i = 0
        while i < steps_num:

            time_step = dt[i]
            normals = normal_draws[i]

            if time_step > tolerance:
                next_vol = self._update_variance(
                    mean_reversion[i], theta[i], volvol[i], rho[i], current_vol, time_step, normals[..., 0])
                next_log_spot = self._update_log_spot(
                    mean_reversion[i], theta[i], volvol[i], rho[i], current_vol, next_vol, current_log_spot, time_step, normals[..., 1])
            else:
                next_vol = current_vol
                next_log_spot = current_log_spot

            vol_paths[i] = mnp.asarray(next_vol, dtype=self._dtype)
            log_spot_paths[i] = mnp.asarray(next_log_spot, dtype=self._dtype)

            current_vol = next_vol
            current_log_spot = next_log_spot

            i += 1

        return mnp.stack([log_spot_paths, vol_paths], axis=0)

    def _update_variance(
            self,
            mean_reversion,
            theta, volvol,
            rho,
            current_vol,
            time_step,
            normals,
            psi_c=1.5
    ):
        """update variance one step"""
        psi_c = mnp.asarray(psi_c, dtype=mean_reversion.dtype)
        scaled_time = mnp.exp(-mean_reversion * time_step)
        volvol_squared = volvol ** 2
        m = theta + (current_vol - theta) * scaled_time
        s_squared = (
            current_vol * volvol_squared * scaled_time / mean_reversion
            * (1 - scaled_time) + theta * volvol_squared / 2 / mean_reversion
            * (1 - scaled_time)**2
        )

        psi = s_squared / m**2
        uniforms = 0.5 * \
            (1 + P.Erf()(normals / mnp.sqrt(2.0, dtype=mnp.float32)))
        cond = psi < psi_c

        psi_inv = 2 / psi
        b_squared = psi_inv - 1 + mnp.sqrt(psi_inv * (psi_inv - 1))

        a = m / (1 + b_squared)
        next_var_true = a * (mnp.sqrt(b_squared) + mnp.squeeze(normals)) ** 2

        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / m
        next_var_false = mnp.where(uniforms > p, mnp.log(1 - p) - mnp.log(1 - uniforms),
                                   mnp.zeros_like(uniforms)) / beta

        next_vol = mnp.where(cond, next_var_true, next_var_false)

        return next_vol

    def _update_log_spot(
        self,
        mean_reversion,
        theta,
        volvol,
        rho,
        current_vol,
        next_vol,
        current_log_spot,
        time_step,
        normals,
        gamma_1=0.5,
        gamma_2=0.5
    ):
        """update log spot one step"""
        k_0 = - rho * mean_reversion * theta / volvol * time_step
        k_1 = (gamma_1 * time_step
               * (mean_reversion * rho / volvol - 0.5)
               - rho / volvol)
        k_2 = (gamma_2 * time_step
               * (mean_reversion * rho / volvol - 0.5)
               + rho / volvol)
        k_3 = gamma_1 * time_step * (1 - rho ** 2)
        k_4 = gamma_2 * time_step * (1 - rho ** 2)

        next_log_spot = current_log_spot + k_0 + k_1 * current_vol + k_2 * \
            next_vol + mnp.sqrt(k_3 * current_vol + k_4 * next_vol) * normals

        return next_log_spot

    def _get_parameters(self, times, *params):

        result = []
        for param in params:
            if P.IsInstance()(param, P.typeof(piecewise.PiecewiseConstantFunction)):
                result.append(param(times))
            else:
                result.append(param * mnp.ones_like(times))
        return result
