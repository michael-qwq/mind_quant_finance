import mindspore.numpy as mnp
import mindspore.ops as P
from mind_quant_finance.math import piecewise
import mindspore.nn as nn


class HestonModel(nn.Cell):
    
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
        
    def construct(self,
                  initial_state,
                  num_timesteps,
                  num_paths=1,
                  seed=None,
                  tolerance=1e-6,
                  times=None):


        times = mnp.asarray(times, dtype=self._dtype)

        normal_draws = mnp.randn(
            (num_timesteps, num_paths, self._dim), dtype=self._dtype)  # 365 x paths x 2

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

    def _sample_path(self,
                     times,
                     current_log_spot,
                     current_vol,
                     num_paths,
                     tolerance,
                     normal_draws):

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
            mean_reversion, theta, volvol, rho, current_vol, time_step, normals, psi_c=1.5):
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

    def _update_log_spot(self, mean_reversion, theta, volvol, rho,
                         current_vol, next_vol, current_log_spot, time_step, normals,
                         gamma_1=0.5, gamma_2=0.5):

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