import mindspore.numpy as np
from mindspore import dtype as mstype


def get_discounted_rate_factor(discounted_factors, discounted_rates, shape,
                             expiries, dtype=mstype.float32):
    r"""
    Conversion between discounted rate and discounted factor

    Inputs:
        - **discounted_factors** (Tensor) - A ms.Tensor with the same shape of der_price/ spot.
        - **discounted_rates** (Tensor) - A ms.Tensor with the same shape of der_price/spot.
                                          The discount rate of the der_price/spot. (r)
                                          At most one of the factor/rate can be supplied.
                                          If both are None, it means discount_rate = 1.0 (Default/No discount).
        - **shape** (tuple) - Shape which has define the shape of discount_factor/discount_rate.
        - **expiries** (Tensor) - The expiry of the product.
        - **dtype** (mstype) - The data type of function. Default mstype.float32.

    Outputs:
        - **factors** (Tensor) - The corresponding discount_factors / discount_rate.
        - **rates** (Tensor) - The corresponding discount_rates.
    """
    if (discounted_rates is not None) and (discounted_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may '
                         'be supplied')
    elif (discounted_factors is None) and (discounted_rates is not None):
        discounted_factors = np.exp(-discounted_rates * expiries)
    elif (discounted_rates is None) and (discounted_factors is not None):
        discounted_rates = -np.log(discounted_factors) / expiries
    else:
        discounted_rates = np.zeros(shape, dtype=dtype)
        discounted_factors = np.ones(shape, dtype=dtype)
    return discounted_factors, discounted_rates
