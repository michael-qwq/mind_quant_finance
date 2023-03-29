"Generic Ito Process"

from typing import Callable, Optional, Any

import mindspore
import mindspore.ops as P
import mindspore.numpy as np
from mindspore import Tensor

import mindspore.nn as nn
import mindspore.common.dtype as mstype

from mind_quant_finance.math.ops import matvec

from mind_quant_finance.math.random_ops import random


Dtype = Any

class MCGenericltoProcess(nn.Cell):
    def __init__(self, 
        dim: int, 
        dirft_fn: Callable, 
        volatility_fn: Callable,
        random_type: Optional[random.RandomType] = random.RandomType.PSEUDO,
        dtype: Dtype = mstype.float32):
        
        super(MCGenericltoProcess, self).__init__()

        self.dim = dim
        self.drift_fn = dirft_fn
        self.volatility_fn = volatility_fn
        self.dtype=dtype

        self.random_ops = random.get_random_ops(random_type)
        self.zeros = P.Zeros()
        self.scalar_to_tensor = P.ScalarToTensor()
        self.sqrt = P.Sqrt()
        

    def construct(self, initial_state, batch_size, num_paths, num_timesteps, dt, times=None):
        
        # normal_draws is a genernated by the random ops
        normal_draws = self.random_ops(batch_size, num_timesteps, num_paths, self.dim, self.dtype)

        current_state = np.broadcast_to(initial_state, (batch_size, num_paths, self.dim))
        result = self.zeros((batch_size, times.shape[0], num_paths, self.dim), self.dtype)
        
        num_timesteps = Tensor(num_timesteps, mindspore.int32)
        i = self.scalar_to_tensor(0, mindspore.int32)
        j = self.scalar_to_tensor(0, mindspore.int32)
        dt = Tensor(dt, dtype=self.dtype)
        
        # To avoid loop unroll, `i` and `num_timesteps` must be Tensor
        while i <= num_timesteps:
            
            dw = normal_draws[:, i, :, :]
            current_time = i * dt
            dt_inc = self.drift_fn(current_time, current_state) * dt
            dw_inc = self.volatility_fn(current_time,current_state) * dw * self.sqrt(dt)
            next_state = current_state + dt_inc + dw_inc
            current_state = next_state
            
            if np.isclose(current_time, times[j], atol=1e-08):
                result[:, j, :, :] = current_state
                j += 1
            i += 1
        
        return result