import mindspore.numpy as np
from mind_quant_finance.engine.analytic.european_option import AnalyticBlackScholesMerton
import mindspore as ms
from mindspore import dtype as mstype
import pytest

dtype = mstype.float32


class TestBSMOptionPrice:

    def test_option_prices(self):
        """Tests that the BS prices are correct."""
        forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)
        volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4], dtype=dtype)
        expiries = ms.Tensor(1.0, dtype=dtype)
        expected_prices = np.array(
            [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933], dtype=dtype)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_price_zero_vol(self):
        """Tests that zero volatility is handled correctly."""
        # If the volatility is zero, the option's value should be correct.
        forwards = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.1, 0.9, 1.1, 0.9], dtype=dtype)
        volatilities = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
        expiries = ms.Tensor(1.0, dtype=dtype)
        is_call_options = np.array([True, True, False, False])
        expected_prices = np.array([0.0, 0.1, 0.1, 0.0], dtype=dtype)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_price_zero_expiry(self):
        """Tests that zero expiry is correctly handled."""
        # If the expiry is zero, the option's value should be correct.
        forwards = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.1, 0.9, 1.1, 0.9], dtype=dtype)
        volatilities = np.array([0.1, 0.2, 0.5, 0.9], dtype=dtype)
        expiries = ms.Tensor(0.0, dtype=dtype)
        is_call_options = np.array([True, True, False, False])
        expected_prices = np.array([0.0, 0.1, 0.1, 0.0], dtype=dtype)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_price_long_expiry_calls(self):
        """Tests that very long expiry call option behaves like the asset."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.1, 0.9, 1.1, 0.9], dtype=dtype)
        volatilities = np.array([0.1, 0.2, 0.5, 0.9], dtype=dtype)
        expiries = ms.Tensor(1e10, dtype=dtype)
        expected_prices = forwards
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_price_long_expiry_puts(self):
        """Tests that very long expiry put option is worth the strike."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([0.1, 10.0, 3.0, 0.0001], dtype=dtype)
        volatilities = np.array([0.1, 0.2, 0.5, 0.9], dtype=dtype)
        expiries = ms.Tensor(1e10)
        expected_prices = strikes
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=False)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_price_vol_and_expiry_scaling(self):
        """Tests that the price is invariant under vol->k vol, T->T/k**2."""
        n = 20
        forwards = np.exp(np.randn(n), dtype=dtype)
        volatilities = np.exp(np.randn(n, dtype=dtype) / 2, dtype=dtype)
        strikes = np.exp(np.randn(n, dtype=dtype), dtype=dtype)
        expiries = np.exp(np.randn(n, dtype=dtype), dtype=dtype)
        scaling = ms.Tensor(5.0, dtype=dtype)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32)
        base_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        scaled_prices = solver(
            expiries=expiries / scaling / scaling,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities * scaling)
        print(base_prices)
        print(scaled_prices)
        assert (np.abs(base_prices - scaled_prices) < 1e-6).all()

    def test_option_prices_detailed_discount(self):
        """Tests the prices with discount_rates."""
        spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0] * 2, dtype=dtype)
        strikes = np.array([100.0] * 10, dtype=dtype)
        discount_rates = ms.Tensor(0.08, dtype=dtype)
        volatilities = ms.Tensor(0.2, dtype=dtype)
        expiries = ms.Tensor(0.25, dtype=dtype)
        is_call_options = np.array([True] * 5 + [False] * 5)
        dividend_rates = ms.Tensor(0.12, dtype=dtype)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=spots,
            volatilities=volatilities,
            discounted_rates=discount_rates,
            dividend_rates=dividend_rates)
        expected_prices = np.array(
            [0.03, 0.57, 3.42, 9.85, 18.62, 20.41, 11.25, 4.40, 1.12, 0.18])
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 5e-3).all()

    def test_bachelier_positive_underlying(self):
        """Tests that the bachelier with positive prices are correct."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0], dtype=dtype)
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        expected_prices = np.array([
            0.3989423, 0.0833155, 0.3989423, 0.1977966, 0.2820948, 0.5641896
        ], dtype=dtype)
        solver = AnalyticBlackScholesMerton(is_normal_volatility=True, dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_bachelier_negative_underlying(self):
        """Tests that the bachelier with negative prices are correct."""
        forwards = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=dtype)
        strikes = np.array([1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -0.5], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0], dtype=dtype)
        is_call_options = np.array([True, True, True, True, True, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        expected_prices = np.array([
            0.0084907, 0.0833155, 0.3989423, 0.5641896, 0.2820948, 0.3989423,
            0.6977965
        ], dtype=dtype)
        solver = AnalyticBlackScholesMerton(is_normal_volatility=True, dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_bachelier_at_the_money(self):
        """  bachelier model, these are the cases for at the money (forward = strike)."""
        forwards = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0], dtype=dtype)
        strikes = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0], dtype=dtype)
        is_call_options = np.array([True, True, True, False, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        expected_prices = np.array([
            0.3989423, 0.3989423, 0.5641896, 0.3989423, 0.3989423, 0.5641896
        ], dtype=dtype)
        solver = AnalyticBlackScholesMerton(is_normal_volatility=True, dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()

    def test_bachelier_tricky(self):
        """Tests the Newton root finder recovers the volatility on a few cases."""
        forwards = np.array([0.00982430235191995])
        strikes = np.array([0.00982430235191995])
        expiries = np.array([0.5])
        is_call_options = np.array([True])
        volatilities = np.array([0.01])
        expected_prices = np.array([0.002820947917738782])
        solver = AnalyticBlackScholesMerton(is_normal_volatility=True, dtype=mstype.float32, is_call_options=is_call_options)
        computed_prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities)
        print(expected_prices)
        print(computed_prices)
        assert (np.abs(computed_prices - expected_prices) < 1e-6).all()


if __name__ == '__main__':
    pytest.main('test_bsm_option_price.py')
