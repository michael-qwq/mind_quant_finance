from typing import Optional
import enum

import mindspore as ms
import mindspore.numpy as np
from mindspore import dtype as mstype
from mindspore import ms_function

from mind_quant_finance.math.root_search import newton
from mind_quant_finance.utils.rate_factor import get_discounted_rate_factor
from mind_quant_finance.math.distribution import standnormal_pdf, standnormal_cdf


@enum.unique
class ImpliedVolUnderlyingDistribution(enum.Enum):
    r"""Underlying distribution.
      * `BSM`: Lognormal distribution for the standrad Black-Scholes models.
      * `Bachelier`: Normal distribution used in Bachelier model.
    """
    BSM = 0
    Bachelier = 1


def implied_vol_solver(expiries, strikes, der_prices, spots,
                       discounted_factors=None,
                       discounted_rates=None,
                       is_call_options=None,
                       underlying_distribution=ImpliedVolUnderlyingDistribution.BSM,
                       initial_volatilities=None,
                       tolerance: Optional[float] = 1e-6,
                       max_iterations: Optional[int] = 20,
                       dtype=mstype.float32
                       ):
    r"""
    Calculate the implied volatilities based on BSM formula

    Inputs:
        - **expiries** (Tensor) - The expiry of the options.
        - **strikes** (Tensor) - The strikes of the options.
        - **der_prices** (Tensor) - The current price of the options.
        - **spots** (Tensor) - The spot of the underlying product.
        - **discounted_factors** (Tensor) - A ms.Tensor with the same shape of der_price/ spot.
                                            The discount factor of the der_price/spot. (e^{-rT}).
                                            Default: None.
        - **discounted_rates** (Tensor) - A ms.Tensor with the same shape of der_price/spot.
                                          The discount rate of the der_price/spote. (r)
                                          discount_factor = exp(-expiries*discount_rate).
                                          At most one of the factor/rate can be supplied.
                                          If both are None, it means discount_rate = 1.0 (Default/No discount).
                                          Default: None.
        - **is_call_options** (Tensor) - A boolean ms.Tensor of a shape compatible with `prices`.
                                         Indicates whether the option is a call (True) or a put (False).
                                         If not supplied, call options are assumed.
                                         Default: None.
        - **initial_volatilities** (Tensor) - the initial point of volatilities for Implied-Vol Solver.
                                              if None, the algorithm will initialize it automatically.
                                              Default: None.
        - **underlying_distribution** (enum) - Enum value of ImpliedVolUnderlyingDistribution.
                                               Select the distribution of the underlying.
                                               (BSM / Bachelier Model)
                                               Default: ImpliedVolUnderlyingDistribution.BSM
        - **tolerance** (float) - The root finder will stop where this tolerance is crossed. Default: 1e-6.
        - **max_iterations** (int) - The maximum number of iterations of Implied-Vol Solver. Default: 20.
        - **dtype** (mstype): The data type of the solver. Default: mstype.float32.
    
    Outputs:
        - **implied_vol result** (tuple) -
            A tuple with content of Tuple(vols, converged, failed)
            vols: the implied volatilities given by Implied Vol Solver.
            converged: indicating whether the corresponding root results in an objective
                       function value less than the tolerance.
            failed: indicating whether the corresponding 'root' is not finite.
            root not converged & not failed: limit by the max_iterations, 
                        not fail but also not converged, need to increase the max_iterations
                        to get the needy tolerance / accuracy.
    """
    discounted_factors, discounted_rates = \
        get_discounted_rate_factor(discounted_factors, discounted_rates, der_prices.shape,
                                   expiries, dtype)
    der_prices = der_prices / discounted_factors
    obj_prices = spots / discounted_factors
    if initial_volatilities is None:
        initial_volatilities = der_prices * np.sqrt(2.0 * np.pi)
    if is_call_options is None:
        is_call_options = np.ones_like(der_prices, dtype=mstype.bool_)
    strikes_abs = np.abs(strikes)
    obj_price_abs = np.abs(obj_prices)
    normalized_mask = ms.Tensor(strikes_abs > obj_price_abs, mstype.bool_)
    normalization = np.where(normalized_mask, strikes_abs, obj_price_abs)
    normalization = np.where(np.equal(normalization, np.zeros(1)),
                             np.ones_like(normalization), normalization)
    der_prices = der_prices / normalization
    obj_prices = obj_prices / normalization
    strikes = strikes / normalization

    def _bsm_lognormal_vega_func():
        lnf = np.log(obj_prices) - np.log(strikes)
        sqrt_t = np.sqrt(expiries)

        @ms_function
        def val_vega_func(volatilities):
            vol_t = volatilities * sqrt_t
            d1 = (lnf / vol_t + vol_t / 2)
            d2 = d1 - vol_t
            implied_call = obj_prices * standnormal_cdf(d1) - strikes * standnormal_cdf(d2)
            implied_put = implied_call - obj_prices + strikes
            implied_prices = np.where(is_call_options, implied_call, implied_put)
            vega = obj_prices * standnormal_pdf(d1) * sqrt_t / discounted_factors
            return implied_prices - der_prices, vega

        return val_vega_func

    # Ref: https://optionsformulas.com/pages/bachelier-with-drift-delta-gamma-and-vega-derivation.html
    def _bachelier_normal_vega_func():
        sqrt_t = np.sqrt(expiries)

        @ms_function
        def val_vega_func(volatilities):
            vol_t = volatilities * sqrt_t / normalization
            d1 = (obj_prices - strikes) / vol_t
            implied_call = (obj_prices - strikes) * standnormal_cdf(d1) + vol_t * standnormal_pdf(d1)
            implied_put = implied_call - obj_prices + strikes
            implied_price = np.where(is_call_options, implied_call, implied_put)
            vega = sqrt_t * standnormal_pdf(d1) / discounted_factors / normalization
            return implied_price - der_prices, vega

        return val_vega_func

    if underlying_distribution is ImpliedVolUnderlyingDistribution.BSM:
        func_cell = _bsm_lognormal_vega_func()
    elif underlying_distribution is ImpliedVolUnderlyingDistribution.Bachelier:
        func_cell = _bachelier_normal_vega_func()
    else:
        raise AttributeError("The Underlying Distribution of ImpliedVol is not Supported.")
    result = newton.newton_root_finder(func_cell, initial_volatilities,
                                       max_iterations=max_iterations, tolerance=tolerance
                                       )
    return result
