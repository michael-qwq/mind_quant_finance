from typing import Callable, Optional, List, Union

import numpy as np

import mindspore
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore import ms_function

import random_sampler
from random_sampler import RandomType
from ..math.ops import matvec


def sample(dim: int,
           drift_fn: Callable[..., Tensor],
           volatility_fn: Callable[..., Tensor],
           times: Union[int, float] = None,
           time_step: float = None,
           num_time_steps: int = None,
           num_samples: int = 1,
           initial_state: Tensor = None,
           random_type: Optional[RandomType] = None,
           seed: Optional[int] = None,
           swap_memory: bool = True,
           skip: int = 0,
           precompute_normal_draws: bool = False,
           times_grid: Optional[Tensor] = None,
           normal_draws: Optional[Tensor] = None,
           tolerance=None,
           dtype=mindspore.float32) -> Tensor:

    times = P.Concat(axis=-1)([
        mnp.arange(start=0.0, stop=times, step=time_step, dtype=dtype),
        Tensor([times], dtype)
    ])

    if tolerance is None:
        tolerance = 1e-6 if dtype == mindspore.float32 else 1e-3
    tolerance = Tensor(tolerance, dtype)

    if initial_state is None:
        initial_state = P.Zeros()(dim, dtype)

    if not isinstance(initial_state, Tensor):
        initial_state = Tensor(initial_state, dtype)

    batch_shape = initial_state.shape[:-2]
    num_requested_times = times.shape[0]

    if normal_draws is not None:
        normal_draws = Tensor(normal_draws, dtype)

    return _sample(dim=dim,
                   batch_shape=batch_shape,
                   drift_fn=drift_fn,
                   volatility_fn=volatility_fn,
                   times=times,
                   keep_mask=None,
                   num_requested_times=num_requested_times,
                   num_samples=num_samples,
                   initial_state=initial_state,
                   random_type=random_type,
                   seed=seed,
                   swap_memory=swap_memory,
                   skip=skip,
                   precompute_normal_draws=precompute_normal_draws,
                   normal_draws=normal_draws,
                   dtype=dtype)


def _sample(dim,
            batch_shape,
            drift_fn,
            volatility_fn,
            times,
            keep_mask,
            num_requested_times,
            num_samples,
            initial_state,
            random_type,
            seed,
            swap_memory,
            skip,
            precompute_normal_draws,
            normal_draws,
            dtype=mindspore.float32):
    dt = times[1:] - times[:-1]
    sqrt_dt = P.Sqrt()(dt)
    steps_num = dt.shape[-1]
    # for one option, do `num_samples` samples
    current_state = initial_state + P.Zeros()((num_samples, dim), dtype)

    wiener_mean = None

    if normal_draws is None:
        if precompute_normal_draws or random_type in (RandomType.SOBOL,
                                                      RandomType.HALTON):
            normal_draws = random_sampler.generate_mc_normal_draws(
                dim=dim,
                num_time_steps=steps_num,
                num_samples=num_samples,
                batch_shape=batch_shape,
                random_type=random_type,
                skip=skip,
                dtype=dtype,
                seed=seed)
            wiener_mean = None
        else:
            # If pseudo or anthithetic sampling is used, do random sample at each step.
            wiener_mean = np.zeros((dim, ), dtype=np.float32)
            normal_draws = None

    result = current_state
    
    for i in range(steps_num):
        result = _euler_step(i=i,
                             current_state=result,
                             drift_fn=drift_fn,
                             volatility_fn=volatility_fn,
                             wiener_mean=wiener_mean,
                             num_samples=num_samples,
                             times=times,
                             dt=dt,
                             sqrt_dt=sqrt_dt,
                             keep_mask=keep_mask,
                             random_type=random_type,
                             seed=seed,
                             normal_draws=normal_draws)

    return result

def _euler_step(i, current_state, drift_fn, volatility_fn, wiener_mean,
                num_samples, times, dt, sqrt_dt, keep_mask, random_type, seed,
                normal_draws):
    """Performs one step of Euler scheme."""
    current_time = times[i + 1]

    if normal_draws is not None:
        dw = normal_draws[i]
    else:
        dw = random_sampler.multivariate_normal((num_samples, ),
                                                mean=wiener_mean,
                                                random_type=random_type,
                                                seed=seed)
    dw = dw * sqrt_dt[i]
    dt_inc = dt[i] * drift_fn(current_time, current_state)

    dw_inc = matvec(volatility_fn(current_time, current_state), dw)
    next_state = current_state + dt_inc + dw_inc

    result = next_state

    return result
