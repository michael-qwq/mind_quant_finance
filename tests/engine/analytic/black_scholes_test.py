import time
import argparse
import mindspore
import mindspore.context as context
import mindspore.numpy as mnp
from mindspore import ms_function

import vanilla_prices


def test_option_prices1():
    """Tests that the BS prices are correct."""
    dtype = mindspore.float32
    forwards = mnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    strikes = mnp.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
    volatilities = mnp.array([0.0001, 102.0, 2.0, 0.1, 0.4], dtype=dtype)
    expiries = 1.0
    computed_prices = vanilla_prices.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        dtype=dtype,
    )
    expected_prices = mnp.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933], dtype=dtype
    )

    isclose = mnp.isclose(computed_prices, expected_prices, 1e-6)
    return isclose


def test_option_prices2():
    """Tests that the BS prices are correct."""
    rate = 0.05
    expiries = mnp.array([0.5, 1.0, 2.0, 1.3])
    discount_factors = mnp.exp(-rate * expiries)
    # Current value of assets.
    spots = mnp.array([0.9, 1.0, 1.1, 0.9])
    # Forward value of assets at expiry.
    forwards = spots / discount_factors
    # Strike prices given by:
    strikes = mnp.array([1.0, 2.0, 1.0, 0.5])
    # Indicate whether options are call (True) or put (False)
    is_call_options = mnp.array([True, True, False, False])
    # The volatilites at which the options are to be priced.
    volatilities = mnp.array([0.7, 1.1, 2.0, 0.5])

    computed_prices = vanilla_prices.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        discount_factors=discount_factors,
        is_call_options=is_call_options,
        dtype=mindspore.float32,
    )

    expected_prices = mnp.array([0.14798729, 0.24216815, 0.74814549, 0.02260333])

    isclose = mnp.isclose(computed_prices, expected_prices, 1e-5)
    print(f"is correct {isclose}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="black sholes options test")
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
        default=True,
        type=bool,
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
    test_option_prices1()
    end = time.time()
    print(f"#1 time {end - start}")

    start = time.time()
    test_option_prices1()
    end = time.time()
    print(f"#2 time {end - start}")

    start = time.time()
    test_option_prices2()
    end = time.time()
    print(f"#1 time {end - start}")

    start = time.time()
    test_option_prices2()
    end = time.time()
    print(f"#2 time {end - start}")
