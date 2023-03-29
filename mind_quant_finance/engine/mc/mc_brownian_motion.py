from typing import Optional

import mindspore as ms
import mindspore.numpy as np
from mindspore import nn, ops
from mindspore import ms_function
from mindspore import dtype as mstype

from mind_quant_finance.math.ops import cholesky
from mind_quant_finance.math.random_ops.random import generate_mc_normal_draws


class McBrownianModel(nn.Cell):

    def __init__(self, means, volatilities, corr_matrix=None, dtype=mstype.float32):
        r"""
        Mento-Carlo Simulation for High Dimension Brownian Motion

        Args:
            - **mean** (Tensor or float) - Long run means, a ms.Tensor with shape=(batch,dim)/(dim,) or float.
            - **volatilities** (Tensor or float) - Volatilities, a ms.Tensor with shape=(batch,dim)/(dim,) or float.
            - **corr_matrix** (Tensor) - A ms.Tensor with shape=(batch, dim, dim)/(dim, dim),
                                         if None, means that there is no correlation between each dimension.
                                         Default: None.
            - **dtype** (mstype) - The data type of the model. Default: mstype.float32

        Inputs:
            - **times** (Tensor) - Timestamp sequence, a tensor with shape=(batch, n_times) or (n_times,).
                                   No need for times[0] = 0.
            - **initial_state** (Tensor or float) - initial state, a tensor[shape=(batch, dim) / (dim,)] or float.
                                                    Default: 1.0.
            - **num_sample** (int) - The amount of sample path for each batch. Default: 1.
            - **random_type** (int) - Types of random number sequences.
                                      int[Enum.value] from mind_quant_finance.math.random_ops.
                                      Default: 0(RandomType.PSEUDO).
            - **seed** (int) - Random seed for mento-carlo. Default: 0.

        Outputs:
            - **mc_brownian_path** (Tensor) - Mento-carlo result, tensor with shape=(batch, num_sample, n_times+1, dim).
                                              Notice that value = r0 with timestamp = 0.
        """
        super(McBrownianModel, self).__init__()
        self.dtype = dtype
        if type(means) == float and type(volatilities) == float:
            dim = 1
        else:
            dim = means.shape[-1] if type(volatilities) == float else volatilities.shape[-1]
        if type(means) == float:
            means = ms.Tensor([means])
        if type(volatilities) == float:
            volatilities = ms.Tensor([volatilities])
        if means.ndim == 1:
            means = np.expand_dims(means, axis=0)
        if volatilities.ndim == 1:
            volatilities = np.expand_dims(volatilities, axis=0)

        def _expand2(x):
            return np.expand_dims(np.expand_dims(x, 1), 1)

        self._means = _expand2(means)
        self._volatilities = _expand2(volatilities)  # (batch, 1, 1, dim)
        self._volatilities_squared = self._volatilities ** 2
        self.cumsum_ops = ops.CumSum()
        self.dim = dim
        self.batch = self._means.shape[0]

        @ms_function
        def _corr_pass(w):
            return w

        @ms_function
        def _corr_process(w):
            # (batch, 1, 1, dim, dim) * (batch, num_path, times, dim, 1)
            return np.squeeze(np.matmul(cholesky_matrix, np.expand_dims(w, -1)), -1)

        if corr_matrix is None:
            self.corr_func = _corr_pass
        else:
            if corr_matrix.ndim == 2:
                corr_matrix = np.repeat(np.expand_dims(corr_matrix, 0), self.batch, 0)
            cholesky_matrix = _expand2(cholesky(corr_matrix))
            self._cholesky_matrix = cholesky_matrix
            self.corr_func = _corr_process
        self.init_shape = np.zeros((self.batch, self.dim))

    def _squeeze2(self, x):
        return np.squeeze(np.squeeze(x, 1), 1)

    def means(self):
        """
        Return the means of brownian motion.

        Output:
            Tensor, the means of brownian motion.
        """
        return self._squeeze2(self._means)

    def volatilities(self):
        """
        Return the volatilities of brownian motion.

        Output:
            Tensor, the volatilities of brownian motion.
        """
        return self._squeeze2(self._volatilities)

    def cholesky_matrix(self):
        """
        Return the cholesky decomposition of correlation matrix of brownian motion.

        Output:
            Tensor, the cholesky decomposition of correlation matrix of brownian motion.
        """
        return self._squeeze2(self._cholesky_matrix)

    def corr_matrix(self):
        """
        Return the correlation matrix of brownian motion.

        Output:
            Tensor, the correlation matrix of brownian motion.
        """
        _choles = self.cholesky_matrix()
        _ein = ops.Einsum('ijk->ikj')
        return ops.matmul(_ein((_choles,)), _choles)

    def construct(self,
                  times: ms.Tensor,
                  initial_state=1.0,
                  num_sample: int = 1,
                  random_type: Optional[int] = 0,
                  seed: Optional[int] = 0):
        times = np.zeros((self.batch, 1)) + times  # (batch, n_times)
        n_times = times.shape[-1]
        element_shape = np.zeros((self.batch, num_sample, n_times, self.dim))
        times = np.concatenate([np.zeros((self.batch, 1)), times], axis=1)
        dt = times[:, 1:] - times[:, :-1]  # (batch, n_times)
        dt = np.expand_dims(np.repeat(np.expand_dims(dt, 1), num_sample, 1), -1)
        dt = dt + element_shape  # (batch, num_sample, n_times-1, dim)

        dw = generate_mc_normal_draws(element_shape.shape, random_type, seed, self.dtype)
        dw = self.corr_func(dw)
        log_increment = dt * (self._means - self._volatilities_squared * 0.5) + dw * np.sqrt(dt) * self._volatilities
        log_increment = self.cumsum_ops(log_increment, -2)
        increment = np.exp(log_increment)

        initial_state = initial_state + self.init_shape
        initial_state = np.expand_dims(np.expand_dims(initial_state, 1), 1)
        mc_brownian_path = np.ones((self.batch, num_sample, n_times + 1, self.dim)) * initial_state
        mc_brownian_path[:, :, 1:, :] = mc_brownian_path[:, :, 1:, :] * increment
        return mc_brownian_path
