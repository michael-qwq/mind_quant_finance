import mindspore
import numpy as np
import mindspore.numpy as mnp
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor


def option_price(volatilities: Tensor,
                 strikes: Tensor,
                 expiries: Tensor,
                 spots: Tensor = None,
                 forwards: Tensor = None,
                 discount_rates: Tensor = None,
                 dividend_rates: Tensor = None,
                 discount_factors: Tensor = None,
                 is_call_options: Tensor = None,
                 is_normal_volatility: bool = False,
                 dtype: mindspore.dtype = mindspore.float32) -> Tensor:
    """Computes the Black Scholes price for a batch of call or put options.
    """
    if (spots is None) == (forwards is None):
        raise ValueError(
            'Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError(
            'At most one of discount_rates and discount_factors may be supplied')
    if discount_rates is not None:
        discount_factors = mnp.exp(-discount_rates * expiries)
    elif discount_factors is not None:
        discount_rates = mnp.log(discount_factors) / expiries
    else:
        discount_rates = Tensor(0.0, dtype=dtype)
        discount_factors = Tensor(1.0, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = Tensor(0.0, dtype=dtype)

    if forwards is None:
        forwards = spots * \
            mnp.exp((discount_rates - dividend_rates) * expiries)

    sqrt_var = volatilities * mnp.sqrt(expiries)
    nd = msd.Normal(0.0, 1.0, dtype=dtype)
    # lognormal model
    if not is_normal_volatility:
        d1 = mnp.divide(mnp.log(forwards / strikes),
                        sqrt_var) + sqrt_var / 2
        d2 = d1 - sqrt_var

        undiscounted_calls = mnp.where(sqrt_var > 0,
                                       forwards *
                                       nd.cdf(d1) - strikes * nd.cdf(d2),
                                       mnp.maximum(forwards - strikes, 0.0))
    else:  # normal model
        d1 = mnp.divide((forwards - strikes), sqrt_var)
        undiscounted_calls = mnp.where(
            sqrt_var > 0.0,
            (forwards - strikes) * nd.cdf(d1) + sqrt_var *
            mnp.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi),
            mnp.maximum(forwards - strikes, 0.0))

    if is_call_options is None:
        return discount_factors * undiscounted_calls

    undiscounted_forward = forwards - strikes
    undiscounted_puts = undiscounted_calls - undiscounted_forward
    return discount_factors * mnp.where(is_call_options, undiscounted_calls,
                                        undiscounted_puts)
