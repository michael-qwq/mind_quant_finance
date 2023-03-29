import mindspore.numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ms_function

from mind_quant_finance.math.distribution import standnormal_pdf, standnormal_cdf


class AnalyticBlackScholesMerton(nn.Cell):
    def __init__(self, is_call_options=True, is_normal_volatility: bool = False, dtype=mstype.float32):
        r"""
        Calculate the Analytical Result of Option Price Based on BSM Formula.

        Args:
            - **is_call_options** (Tensor or bool)- A boolean ms.Tensor of a shape compatible with `prices`.
                                                    Indicates whether the option is a call (True) or a put (False).
                                                    If not supplied, call options are assumed.
                                                    Default: True.
            - **is_normal_volatility** (bool) - An optional Python boolean specifying whether the volatilities
                                                correspond to lognormal Black volatility (if False) or normal
                                                Black volatility (if True). Default: False.
            - **dtype** (mstype) - The data type of the solver. Default: mstype.float32.


        Notice: All parameters and inputs should be able to broadcast to the same shape.

        Inputs:
            - **expiries** (Tensor) - The expiry of the options.
            - **strikes** (Tensor) - The strikes of the options.
            - **spots** (Tensor) - The spots price of the underlying object.
            - **volatilities** (Tensor) - The volatilities of the options.
            - **discounted_rates** (Tensor) - The discounted rate of the der_price/spots.
                                             discounted_factor = exp(-expiries*discounted_rate).
                                             Default: 0.
            - **dividend_rates** (Tensor or float) - The dividend_rates q of the object, Default: 0.

        Outputs:
            - **option_price** (Tensor) - The price of options with shape of broadcast result.
        """
        super(AnalyticBlackScholesMerton, self).__init__()

        @ms_function
        def _bsm_lognormal_option_price_solve(expiries, strikes, obj_prices, volatilities, is_call_options, dtype):
            lnf = np.log(obj_prices) - np.log(strikes)
            sqrt_t = np.sqrt(expiries)
            vol_t = volatilities * sqrt_t
            d1 = (lnf / vol_t + vol_t / 2)
            d2 = d1 - vol_t
            der_call = np.where(vol_t > 0,
                                obj_prices * standnormal_cdf(d1) - strikes * standnormal_cdf(d2),
                                np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                                )
            der_put = der_call - obj_prices + strikes
            der_price_undiscounted = np.where(is_call_options, der_call, der_put)
            return der_price_undiscounted

        @ms_function
        def _bsm_normal_option_price_solve(expiries, strikes, obj_prices, volatilities, is_call_options, dtype):
            sqrt_t = np.sqrt(expiries)
            vol_t = volatilities * sqrt_t
            d1 = np.divide((obj_prices - strikes), vol_t)
            der_call = np.where(vol_t > 0,
                                (obj_prices - strikes) * standnormal_cdf(d1) + vol_t * standnormal_pdf(d1),
                                np.maximum(obj_prices - strikes, np.zeros(1, dtype=dtype))
                                )
            der_put = der_call - obj_prices + strikes
            der_price_undiscounted = np.where(is_call_options, der_call, der_put)
            return der_price_undiscounted

        if is_normal_volatility:
            self.calc = _bsm_normal_option_price_solve
        else:
            self.calc = _bsm_lognormal_option_price_solve
        self.dtype = dtype
        self.is_call_options = is_call_options

    def construct(self, expiries, strikes, spots, volatilities, discounted_rates=0, dividend_rates=0):
        obj_prices = spots * np.exp((discounted_rates - dividend_rates) * expiries)
        forwards = self.calc(expiries, strikes, obj_prices, volatilities, self.is_call_options, self.dtype)
        discounted_factors = np.exp(-discounted_rates * expiries)
        return forwards * discounted_factors
