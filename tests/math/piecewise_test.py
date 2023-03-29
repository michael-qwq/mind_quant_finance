from mind_quant_finance.math import piecewise 
import time
import argparse
import mindspore
import mindspore.context as context
import mindspore.numpy as mnp
from mindspore import ms_function


def test_piecewise_constant_value_no_batch():
    
    for dtype in [mnp.float32]:
        x = mnp.array([0., 0.1, 2. ,11.])
        jump_locations = mnp.array([0.1, 10], dtype=dtype)
        values = mnp.array([3, 4, 5], dtype=dtype)
        piecewise_func = piecewise.PiecewiseConstantFunction(jump_locations, values, dtype=dtype)
        
        
        computed_value = piecewise_func(x)
        expected_value = mnp.array([3., 3., 4., 5.])
        
        isclose = mnp.isclose(computed_value, expected_value, 1e-6)
    return isclose



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="piecewise function test")
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
    print(test_piecewise_constant_value_no_batch())
    end = time.time()
    print(f"#1 time {end - start}")
