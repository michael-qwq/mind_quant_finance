import os

os.environ['GLOG_v'] = '3'
import time
import argparse

import mindspore
import mindspore.ops as P
from mindspore.ops import functional as F
import mindspore.context as context
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

from mind_quant_finance.engine.mc.mc_generic_ito_process import MCGenericltoProcess


def test_sample_paths_1d(use_batch):
    """Tests path properties for 1-dimentional Ito process.

    We construct the following Ito process.

    ````
    dX = mu * sqrt(t) * dt + (a * t + b) dW
    ````

    For this process expected value at time t is x_0 + 2/3 * mu * t^1.5 .
    Args:
      use_batch: Test parameter to specify if we are testing the batch of Euler
        sampling.
      supply_normal_draws: Supply normal draws.
      random_type: `RandomType` of the sampled normal draws.
    """
    dtype = mstype.float32

    def price_eu_options(spot, sigma, strikes):
        rate = Tensor(0.03)

        def drift_fn(t, x):
          return rate - 0.5 * sigma**2
        
        def vol_fn(t, x):
          return np.reshape(sigma, (1, 1))
    
        dim = 1

        process = MCGenericltoProcess(1, drift_fn, vol_fn)
        # times = 0.55
        # num_paths = 200000
        num_paths = 200000
        T = 1.0
        num_timesteps = 100
        dt = T / num_timesteps
        times = Tensor([1.0])
    
        # normal_draws = P.StandardNormal()((1, num_paths // 2, num_timesteps, dim)).astype(dtype)
        # normal_draws = P.Concat(axis=1)([normal_draws, -normal_draws])
       
        log_spot = np.log(spot)
    
        # (batch_size, num_paths, dim)
        start = time.time()
        paths = process(log_spot, 1, num_paths, num_timesteps, dt, times)
        print(f"# 1: {time.time() - start}")

        start = time.time()
        paths = process(log_spot, 1, num_paths, num_timesteps, dt, times)
        print(f"# 2: {time.time() - start}")

        print(np.exp(paths) - strikes)
        
        prices = np.exp(-(rate * T)) * np.mean(P.ReLU()(np.exp(paths) - strikes), axis=2)

        return prices

    sigma = Tensor(0.1)
    spot = Tensor(700, dtype=dtype)

    strikes = Tensor(600, dtype=dtype)

    start = time.time()
    price = price_eu_options(spot, sigma, strikes)
    print(f"# 1: {time.time() - start}")

    print(price)
    
    # expected_means = x0 + (2.0 / 3.0) * mu * np.power(T, 1.5)
    # print(f"means: {means}")
    # print(f"expected means: {expected_means}")



    # print(f"close: {np.isclose(means, expected_means, rtol=1e-2, atol=1e-2)}")
    # print(f"# 1: {time.time() - start}")
    # # paths = paths.asnumpy()
    # # print(paths)

    # start = time.time()
    # paths = process(x0, 1, num_paths, num_timesteps, dt, normal_draws=normal_draws)
    # print(f"# 1: {time.time() - start}")

    # start = time.time()
    # paths = process(x0, 1, num_paths, num_timesteps, dt, normal_draws=None)
    # print(f"# 1: {time.time() - start}")
    # # paths = paths.asnumpy()
    # # print(paths)

    # start = time.time()
    # paths = process(x0, 1, num_paths, num_timesteps, dt, normal_draws=None)
    # print(f"# 1: {time.time() - start}")


context.set_context(
        mode=context.GRAPH_MODE,
        device_target="CPU",
        device_id=0,
        save_graphs=False
)

test_sample_paths_1d(use_batch=False)



    # means = np.mean(paths, axis=-2)
    # expected_means = x0 + (2.0 / 3.0) * mu * np.power(T, 1.5)
    # print(f"means {means}")
    # print(f"expected_means {expected_means}")
    # np.testing.assert_array_almost_equal(means, expected_means, decimal=2)