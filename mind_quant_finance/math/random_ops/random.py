import enum
from functools import reduce

from scipy import stats
from scipy import special

import numpy as onp
import mindspore
import mindspore.ops as P
import mindspore.numpy as np
import mindspore.scipy as mscipy
from mindspore import Tensor
from mindspore import ms_function

from ..ops import matvec

_SQRT_2 = np.sqrt(2.)


@enum.unique
class RandomType(enum.Enum):
    r"""Types of random number sequences.

    * `PSEUDO`: The standard MindSpore random generator.
    * `PSEUDO_ANTITHETIC`: PSEUDO random numbers along with antithetic variates.
    * `HALTON`: The standard Halton sequence.
    * `HALTON_RANDOMIZED`: The randomized Halton sequence.
    * `SOBOL`: The standard Sobol sequence.

    """

    PSEUDO = 0
    PSEUDO_ANTITHETIC = 1
    STATELESS = 2
    STATELESS_ANTITHETIC = 3
    HALTON = 4
    HALTON_RANDOMIZED = 5
    SOBOL = 6


def get_random_ops(random_type):
    if random_type == RandomType.PSEUDO:
        return pseudo


def pseudo(batch_size, num_timesteps, num_paths, dim, dtype):
    return P.StandardNormal()((batch_size, num_timesteps, num_paths, dim)).astype(dtype)


@ms_function
def generate_mc_normal_draws(shape, random_type, seed, dtype=mindspore.float32):
    r"""
    Generates normal random samples to be consumed by a Monte Carlo algorithm.

    Inputs:
        - **shape** (tuple) - The shape of Tensor to generate.
        - **random_type** (int) - Types of random number sequences.
                                  int[Enum.value] from mind_quant_finance.math.random_ops.
        - **seed** (int) - Random seed for Tensor generation.
        - **dtype** (mstype) - The data type of Tensor. Default: mstype.float32.

    Output:
        - **stdnoraml** (Tensor) - Standard normal sample with input shape.
    """
    # (TODO): Current Version only support PSEUDO(0) with dtype=mindspore.float32
    return P.StandardNormal(seed=seed)(shape).astype(dtype)

# def generate_mc_normal_draws(dim,
#                              num_time_steps,
#                              num_samples,
#                              random_type,
#                              batch_shape=None,
#                              skip=0,
#                              seed=None,
#                              dtype=mindspore.float32):
#     """Generates normal random samples to be consumed by a Monte Carlo algorithm.
#     """
#     if skip is None:
#         skip = 0

#     if batch_shape == None:
#         batch_shape = ()
#     # In case of quasi-random draws, the total dimension of the draws should be
#     # `num_time_steps * dim`
#     total_dimension = P.Zeros()((num_time_steps * dim, ), dtype)
#     if random_type in [
#             RandomType.PSEUDO_ANTITHETIC,
#     ]:
#         # Put `num_samples` to the front for antithetic samplers
#         sample_shape = (num_samples, ) + batch_shape
#         is_antithetic = True
#     else:
#         # Note that for QMC sequences `num_samples` should follow `batch_shape`
#         sample_shape = batch_shape + (num_samples, )
#         is_antithetic = False
#     normal_draws = multivariate_normal(sample_shape,
#                                        mean=total_dimension,
#                                        random_type=random_type,
#                                        skip=skip,
#                                        seed=seed)
#     # Reshape and transpose to (batch_shape, num_samples, num_time_steps, dim)
#     normal_draws = normal_draws.reshape(sample_shape + (num_time_steps, dim))
#     # put `num_time_steps` in front
#     # Shape [steps_num] + batch_shape + [num_samples, dim]
#     normal_draws_rank = len(normal_draws.shape)
#     if is_antithetic and normal_draws_rank > 3:
#         # Permutation for the case when the batch_shape is present
#         perm = (normal_draws_rank - 2, ) + tuple(
#             list(range(1, normal_draws_rank - 2))) + (0, normal_draws_rank - 1)
#     else:
#         perm = (normal_draws_rank - 2, ) + tuple(
#             list(range(normal_draws_rank - 2))) + (normal_draws_rank - 1, )
#     normal_draws = P.Transpose()(normal_draws, perm)
#     return normal_draws


# def multivariate_normal(sample_shape,
#                         mean=None,
#                         stddev=None,
#                         covariance_matrix=None,
#                         scale_matrix=None,
#                         random_type=None,
#                         seed=None,
#                         dtype=mindspore.float32,
#                         **kwargs):
#     """Generates draws from a multivariate Normal distribution.

#     Draws samples from the multivariate Normal distribution on `R^k` with the
#     supplied mean and covariance parameters. Allows generating either
#     (pseudo) random or quasi-random draws based on the `random_type` parameter.
#     """
#     random_type = RandomType.PSEUDO if random_type is None else random_type

#     if mean is None and covariance_matrix is None and scale_matrix is None:
#         raise ValueError(
#             "At least one of mean, covariance_matrix or scale_matrix must be specified."
#         )

#     if covariance_matrix is not None and scale_matrix is not None:
#         raise ValueError(
#             "Only one of covariance matrix or scale matrix must be specified")

#     if isinstance(sample_shape, list):
#         sample_shape = tuple(sample_shape)

#     if 'skip' in kwargs:
#         skip = kwargs['skip']
#     else:
#         skip = 0

#     if random_type in [RandomType.PSEUDO]:
#         return _multivarivate_pseudo(sample_shape,
#                                      mean=mean,
#                                      covariance_matrix=covariance_matrix,
#                                      scale_matrix=scale_matrix,
#                                      random_type=random_type,
#                                      seed=seed,
#                                      dtype=dtype)

#     if random_type in [RandomType.PSEUDO_ANTITHETIC]:
#         return _multivariate_pseudo_antithetic(
#             sample_shape,
#             mean=mean,
#             covariance_matrix=covariance_matrix,
#             scale_matrix=scale_matrix,
#             random_type=random_type,
#             seed=seed,
#             dtype=dtype)

#     if random_type in [RandomType.SOBOL, RandomType.HALTON]:
#         return _multivariate_quasi(sample_shape,
#                                    mean=mean,
#                                    random_type=random_type,
#                                    covariance_matrix=covariance_matrix,
#                                    scale_matrix=scale_matrix,
#                                    skip=skip)


# def _process_mean_scale(mean, scale_matrix, covariance_matrix):
#     """Extracts correct mean, scale, batch_shape, dimension, and dtype."""
#     if scale_matrix is None and covariance_matrix is not None:
#         scale_matrix = mscipy.linalg.cholesky(covariance_matrix)
#     if mean is None:
#         mean = 0.0
#         # batch_shape includes the dimension of the samples
#         batch_shape = scale_matrix.shape[:-1]
#         dim = scale_matrix.shape[-1]
#     else:
#         batch_shape = mean.shape
#         dim = mean.shape[-1]
#     return mean, scale_matrix, batch_shape, dim


# def _multivarivate_pseudo(sample_shape,
#                           mean,
#                           covariance_matrix=None,
#                           scale_matrix=None,
#                           random_type: RandomType = RandomType.PSEUDO,
#                           seed=None,
#                           dtype=None):
#     (mean, scale_matrix, batch_shape,
#      dim) = _process_mean_scale(mean, scale_matrix, covariance_matrix)
#     output_shape = sample_shape + batch_shape

#     stdnormal = P.StandardNormal()
#     stdnormal.add_prim_attr("use_curand", True)
#     random_normal = stdnormal(output_shape)
#     normal_mean = Tensor(0.0, mindspore.float32)
#     normal_stddev = Tensor(1.0, mindspore.float32)
#     samples = random_normal * normal_stddev + normal_mean

#     # TODO
#     # before 1.6, there's a bug in normal ops, use code above to genernate normal numbers
#     # samples = P.normal(shape=output_shape,
#     #                    mean=Tensor(0.0, mindspore.float32),
#     #                    stddev=Tensor(1.0, mindspore.float32),
#     #                    seed=seed)

#     if not isinstance(mean, Tensor):
#         mean = Tensor(mean, dtype=dtype)

#     if scale_matrix is not None:
#         scale_matrix = Tensor(scale_matrix, dtype)
#         samples = mean + matvec(scale_matrix, samples)
#     else:
#         samples = mean + samples
#     return samples


# def _multivariate_pseudo_antithetic(sample_shape: tuple,
#                                     mean: np.ndarray,
#                                     random_type: RandomType,
#                                     covariance_matrix: np.ndarray = None,
#                                     scale_matrix: np.ndarray = None,
#                                     seed=None,
#                                     dtype=mindspore.float32):
#     """Returns normal draws with the antithetic samples."""

#     sample_zero_dim = sample_shape[0]
#     # For the antithetic sampler `sample_shape` is split evenly between
#     # samples and their antithetic counterparts. In order to do the splitting
#     # we expect the first dimension of `sample_shape` to be even.
#     if sample_zero_dim % 2 != 0:
#         raise ValueError(
#             "First dimension of `sample_shape` should be even for PSEUDO_ANTITHETIC random type"
#         )
#     antithetic_shape = (sample_zero_dim // 2, ) + sample_shape[1:]
#     random_type_sample = RandomType.PSEUDO

#     result = _multivarivate_pseudo(sample_shape=antithetic_shape,
#                                    mean=mean,
#                                    covariance_matrix=covariance_matrix,
#                                    scale_matrix=scale_matrix,
#                                    random_type=random_type_sample,
#                                    seed=seed,
#                                    dtype=dtype)

#     if mean is None:
#         return P.Concat(axis=0)((result, -result))
#     else:
#         if not isinstance(mean, Tensor):
#             mean = Tensor(mean, dtype)
#         return P.Concat(axis=0)((result, 2 * mean - result))


# def _multivariate_quasi(sample_shape: tuple,
#                         mean: np.ndarray,
#                         random_type: RandomType,
#                         covariance_matrix: np.ndarray = None,
#                         scale_matrix: np.ndarray = None,
#                         dtype=mindspore.float32,
#                         skip: int = 0):
#     (mean, scale_matrix, batch_shape,
#      dim) = _process_mean_scale(mean, scale_matrix, covariance_matrix)

#     # first reverse `batch_shape`, then concat the reversed `batch_shape` and `sample_shape`
#     output_shape = tuple(reversed(batch_shape)) + sample_shape
#     num_samples = reduce((lambda x, y: x * y), output_shape) // dim
#     if random_type == RandomType.SOBOL:
#         low_discrepancy_seq = _sobol(dim=dim,
#                                      num_results=num_samples,
#                                      skip=skip)
#     elif random_type == RandomType.HALTON:
#         low_discrepancy_seq = _halton(dim=dim,
#                                       num_results=num_samples,
#                                       skip=skip)

#     low_discrepancy_seq = low_discrepancy_seq.transpose()

#     samples = special.erfinv((low_discrepancy_seq - 0.5) * 2) * _SQRT_2

#     if not isinstance(samples, Tensor):
#         samples = Tensor(samples, dtype)

#     size_sample_shape = len(sample_shape)
#     size_batch_shape = len(batch_shape)
#     permutation = tuple([
#         i
#         for i in range(size_batch_shape, size_batch_shape + size_sample_shape)
#     ]) + tuple([i for i in range(size_batch_shape - 1, -1, -1)])

#     samples = samples.reshape(output_shape).transpose(permutation)

#     if not isinstance(mean, Tensor):
#         mean = Tensor(mean, dtype=dtype)

#     if scale_matrix is not None:
#         scale_matrix = Tensor(scale_matrix, dtype)
#         samples = mean + matvec(scale_matrix, samples)
#     else:
#         samples = mean + samples

#     return samples


# def _sobol(dim, num_results, skip=0):
#     """Returns draws using the scipy qusai sobol distribution.

#     Return Type: ndarray
#     """
#     if dim < 1:
#         raise ValueError(
#             "Dimension `dim` must be greater than zero. Supplied {}".format(
#                 dim))
#     if num_results < 1:
#         raise ValueError(
#             "Number of results `num_results` must be greater than zero.. Supplied {}"
#             .format(num_results))

#     sampler = stats.qmc.Sobol(d=dim, scramble=False)
#     num_results = num_results + skip + 1
#     samples = sampler.random(num_results)[skip + 1:]

#     return samples


# def _halton(dim, num_results, skip=0):
#     """Returns draws using the scipy qusai halton distribution.

#     Return Type: ndarray
#     """
#     if dim < 1:
#         raise ValueError(
#             "Dimension `dim` must be greater than zero. Supplied {}".format(
#                 dim))
#     if num_results < 1:
#         raise ValueError(
#             "Number of results `num_results` must be greater than zero.. Supplied {}"
#             .format(num_results))

#     sampler = stats.qmc.Halton(d=dim, scramble=False)
#     num_results = num_results + skip + 1
#     samples = sampler.random(num_results)[skip + 1:]

#     return samples

