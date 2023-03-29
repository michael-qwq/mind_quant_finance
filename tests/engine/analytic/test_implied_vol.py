import pytest

import numpy as onp

import mindspore.numpy as np
import mindspore as ms
from mindspore import dtype as mstype

from mind_quant_finance.engine.analytic.implied_vol import implied_vol_solver, ImpliedVolUnderlyingDistribution
from mind_quant_finance.engine.analytic.european_option import AnalyticBlackScholesMerton

dtype = mstype.float32


class TestImpliedVol:

    def test_basic_finder(self):
        """Tests the Newton root finder recovers the volatility on a few cases."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0], dtype=dtype)
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        init_vols = np.array([2.0, 0.5, 2.0, 0.5, 1.5, 1.5], dtype=dtype)
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        prices = np.array([
            0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988
        ], dtype=dtype)
        implied_vols, converged, failed = implied_vol_solver(expiries=expiries,
                                                             strikes=strikes,
                                                             der_prices=prices,
                                                             spots=forwards,
                                                             discounted_factors=discounts,
                                                             is_call_options=is_call_options,
                                                             initial_volatilities=init_vols,
                                                             max_iterations=100)

        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_basic_radiocic_newton_combination_finder(self):
        """Tests the Newton root finder recovers the volatility on a few cases."""
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0], dtype=dtype)
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        prices = np.array([
            0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988
        ], dtype=dtype)
        implied_vols, converged, failed = implied_vol_solver(expiries=expiries,
                                                             strikes=strikes,
                                                             der_prices=prices,
                                                             spots=forwards,
                                                             discounted_factors=discounts,
                                                             is_call_options=is_call_options,
                                                             max_iterations=100)
        assert np.logical_or(converged[:3].all(), converged[4:].all())
        assert failed[3]
        # [yuanchengbo1]
        # Since we use sqrt(2PI)*price to init the newton method,
        # we should not solve the forth point in this test case. (check by hand)
        assert (np.abs(volatilities[:3] - implied_vols[:3]) < 2e-6).all()
        assert (np.abs(volatilities[4:] - implied_vols[4:]) < 2e-6).all()

    def test_bachelier_positive_underlying(self):
        """Tests the Newton root finder recovers the volatility on Bachelier Model.

        This are the cases with positive underlying and strike.
        """
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0], dtype=dtype)
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        init_vols = np.array([2.0, 0.5, 2.0, 1.5, 1.5, 1.5], dtype=dtype)
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        prices = np.array([
            0.3989423, 0.0833155, 0.3989423, 0.1977966, 0.2820948, 0.5641896
        ], dtype=dtype)
        implied_vols, converged, failed = \
            implied_vol_solver(expiries=expiries,
                               strikes=strikes,
                               der_prices=prices,
                               spots=forwards,
                               discounted_factors=discounts,
                               is_call_options=is_call_options,
                               underlying_distribution=ImpliedVolUnderlyingDistribution.Bachelier,
                               initial_volatilities=init_vols,
                               max_iterations=100)

        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_bachelier_negative_underlying(self):
        """Tests the Newton root finder recovers the volatility on Bachelier Model.

        These are the cases with negative underlying and strike.
        """
        forwards = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=dtype)
        strikes = np.array([1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -0.5], dtype=dtype)
        expiries = np.array([1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0], dtype=dtype)
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        init_vols = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        is_call_options = np.array([True, True, True, True, True, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        prices = np.array([
            0.0084907, 0.0833155, 0.3989423, 0.5641896, 0.2820948, 0.3989423,
            0.6977965
        ], dtype=dtype)
        implied_vols, converged, failed = \
            implied_vol_solver(expiries=expiries,
                               strikes=strikes,
                               der_prices=prices,
                               spots=forwards,
                               discounted_factors=discounts,
                               is_call_options=is_call_options,
                               underlying_distribution=ImpliedVolUnderlyingDistribution.Bachelier,
                               initial_volatilities=init_vols,
                               max_iterations=100)
        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_bachelier_at_the_money(self):
        """Tests the Newton root finder recovers the volatility on Bachelier Model.

        These are the cases for at the money (forward = strike).
        """
        forwards = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0], dtype=dtype)
        strikes = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0], dtype=dtype)
        expiries = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0], dtype=dtype)
        init_vols = np.array([2.0, 1.0, 1.0, 2.0, 1.0, 1.0], dtype=dtype)
        is_call_options = np.array([True, True, True, False, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
        prices = np.array([
            0.3989423, 0.3989423, 0.5641896, 0.3989423, 0.3989423, 0.5641896
        ], dtype=dtype)
        implied_vols, converged, failed = \
            implied_vol_solver(expiries=expiries,
                               strikes=strikes,
                               der_prices=prices,
                               spots=forwards,
                               is_call_options=is_call_options,
                               underlying_distribution=ImpliedVolUnderlyingDistribution.Bachelier,
                               initial_volatilities=init_vols,
                               max_iterations=100)
        # print('converged', converged)
        # print('failed', failed)
        # print('implied_vols', implied_vols)
        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_implied_vol_extensive(self):
        num_examples = 1000

        expiries = np.linspace(0.8, 1.2, num_examples, dtype=dtype)
        rates = np.linspace(0.03, 0.08, num_examples, dtype=dtype)
        discounted_factors = np.exp(-rates * expiries, dtype=dtype)
        spots = np.ones(num_examples, dtype=dtype)
        forwards = spots / discounted_factors
        strikes = np.linspace(0.8, 1.2, num_examples, dtype=dtype)
        volatilities = np.ones_like(forwards, dtype=dtype)
        call_options = onp.random.binomial(n=1, p=0.5, size=num_examples)
        is_call_options = ms.Tensor(call_options, dtype=mstype.bool_)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=is_call_options)
        prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=forwards,
            volatilities=volatilities
        )
        implied_vols, converged, failed = implied_vol_solver(expiries=expiries,
                                                             strikes=strikes,
                                                             der_prices=prices,
                                                             spots=forwards,
                                                             is_call_options=is_call_options,
                                                             max_iterations=100)
        assert converged.all()
        assert ~failed.any()
        #         print(volatilities)
        #         print(implied_vols)
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_discount_factor_correctness(self):
        expiries = np.array([1.0], dtype=dtype)
        rates = np.array([0.05], dtype=dtype)
        discounted_factors = np.exp(-rates * expiries, dtype=dtype)
        spots = np.array([1.0], dtype=dtype)
        strikes = np.array([0.9], dtype=dtype)
        volatilities = np.array([0.13], dtype=dtype)
        is_call_options = ms.Tensor(True)
        solver = AnalyticBlackScholesMerton(dtype=mstype.float32, is_call_options=is_call_options)
        prices = solver(
            expiries=expiries,
            strikes=strikes,
            spots=spots,
            volatilities=volatilities,
            discounted_rates=rates
        )

        implied_vols, converged, failed = \
            implied_vol_solver(expiries=expiries,
                               strikes=strikes,
                               der_prices=prices,
                               spots=spots,
                               discounted_factors=discounted_factors,
                               is_call_options=is_call_options,
                               max_iterations=100)
        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()

    def test_bachelier_tricky(self):
        """Tests the Newton root finder recovers the volatility on a few cases."""
        forwards = np.array([0.00982430235191995])
        strikes = np.array([0.00982430235191995])
        expiries = np.array([0.5])
        discounts = np.array([1.0])
        is_call_options = np.array([True])
        volatilities = np.array([0.01])
        prices = np.array([0.002820947917738782])
        implied_vols, converged, failed = \
            implied_vol_solver(expiries=expiries,
                               strikes=strikes,
                               der_prices=prices,
                               spots=forwards,
                               discounted_factors=discounts,
                               is_call_options=is_call_options,
                               underlying_distribution=ImpliedVolUnderlyingDistribution.Bachelier,
                               max_iterations=100)
        assert converged.all()
        assert ~failed.any()
        assert (np.abs(volatilities - implied_vols) < 2e-6).all()


if __name__ == '__main__':
    pytest.main('test_implied_vol.py')
