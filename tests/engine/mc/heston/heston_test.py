from statistics import mean
import numpy as np
from mind_quant_finance.engine.mc.heston import HestonModel
import mindspore.numpy as mnp
import time
from mindspore import Tensor
from mindspore import context
import argparse
from mind_quant_finance.math import piecewise


def test_constant_parameters_heston():

    dtype = mnp.float32
    seed = 1
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
        seed=seed,
        times=times
    )

    return paths


def test_function_parameters_heston():
    
    seed = 1
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
        seed=seed,
        times=times
    )

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mc heson test")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        help="set which type of device you want to use. Ascend/GPU",
    )
    parser.add_argument(
        "--device_id", default=0, type=int, help="device id is for physical devices"
    )
    parser.add_argument(
        "--enable_graph_kernel",
        action='store_true',
        help="whether to use graph kernel",
    )
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
        save_graphs=False,
        enable_graph_kernel=args.enable_graph_kernel,
    )
    start = time.time()
    print(test_constant_parameters_heston())
    end = time.time()
    print(f"#1 time {end - start}")

    start = time.time()
    print(test_function_parameters_heston())
    end = time.time()
    print(f"#2 time {end - start}")
