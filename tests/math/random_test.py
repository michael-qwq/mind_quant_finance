import time
import argparse

import numpy as np
import mindspore
import mindspore.context as context
from mindspore import Tensor
from numpy import random

import mind_quant_finance.math.random as random_sampler
from mind_quant_finance.math.random import RandomType


def test_multivariate_normal_pesudo():
    shape = (10, 2)
    dtype = mindspore.float32
    mean = Tensor(0, dtype)
    stddev = Tensor(1, dtype)
    results = random_sampler.multivariate_normal(
        sample_shape=shape,
        mean=mean,
        stddev=stddev,
        random_type=random_sampler.RandomType.PSEUDO,
        dtype=dtype)
    print(f"test_multivariate_normal_pesudo")
    print(f"pesudo: {results}")


def test_multivariate_normal_sobol():
    shape = (2, 10, 2)
    dtype = mindspore.float32
    results = random_sampler.multivariate_normal(
        sample_shape=shape,
        random_type=random_sampler.RandomType.SOBOL,
        dtype=dtype)

    print(f"test_multivariate_normal_sobol")
    print(f"sobol: {results}")


def test_sobol():
    """quasi random genernator: sobol."""
    samples = random_sampler._sobol(dim=2, num_results=16)

    # print(samples)


def test_halton_numbers_generation():
    """halton random dtype results in the correct draws."""
    dtype = mindspore.float32
    dim = 2
    steps_num = 3
    num_samples = 4
    random_type = RandomType.HALTON
    skip = 10
    samples = random_sampler.generate_mc_normal_draws(dim=dim,
                                                      num_time_steps=steps_num,
                                                      num_samples=num_samples,
                                                      random_type=random_type,
                                                      dtype=dtype,
                                                      skip=skip)
    samples = samples.asnumpy()
    print("test_halton_numbers_generation")
    expected_samples = np.array([[[0.88714653, 0.5350828],
                                  [-0.88714653, -1.0444088],
                                  [0.48877642, -0.04643572],
                                  [-0.15731068, 0.89577985]],
                                 [[-0.5828415, 0.2322723],
                                  [-0.05015358, 0.6270717],
                                  [0.4676988, 1.1628313],
                                  [1.1749868, -1.7412906]],
                                 [[-2.397022, 1.0200763],
                                  [-1.286275, 1.4260769],
                                  [-0.8775918, -2.5170465],
                                  [-0.5798979, -1.3862176]]])
    np.testing.assert_almost_equal(samples, expected_samples, decimal=6)


def test_simple_mean_pseudo():
    """Tests that the sample is correctly generated for pseudo."""
    size = 1000

    mean = np.zeros(size, dtype=np.float32)

    sample = random_sampler.multivariate_normal([size],
                                                mean=mean,
                                                scale_matrix=None,
                                                random_type=RandomType.PSEUDO,
                                                seed=4567)
    sample = sample.asnumpy()
    print("test_general_mean_covariance_pseudo")
    np.testing.assert_array_equal(sample.shape, [size])
    np.testing.assert_array_almost_equal(np.mean(sample, axis=0),
                                         mean,
                                         decimal=1)


def test_general_mean_covariance_pseudo():
    """Tests that the sample is correctly generated for pseudo."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    covariance = np.matmul(scale, scale.transpose())
    size = 30000
    sample = random_sampler.multivariate_normal([size],
                                                mean=mean,
                                                scale_matrix=scale,
                                                random_type=RandomType.PSEUDO,
                                                seed=4567)
    sample = sample.asnumpy()
    print("test_general_mean_covariance_pseudo")
    np.testing.assert_array_equal(sample.shape, [size, 2, 2])
    np.testing.assert_array_almost_equal(np.mean(sample, axis=0),
                                         mean,
                                         decimal=1)
    np.testing.assert_array_almost_equal(np.cov(sample[:, 0, :], rowvar=False),
                                         covariance,
                                         decimal=1)
    np.testing.assert_array_almost_equal(np.cov(sample[:, 1, :], rowvar=False),
                                         covariance,
                                         decimal=1)


def test_mean_and_scale_pseudo_antithetic():
    """Tests antithetic sampler for scale specification."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])

    covariance = np.matmul(scale, scale.transpose())
    size = 30000
    seed = 42
    sample = random_sampler.multivariate_normal(
        [size],
        mean=mean,
        scale_matrix=scale,
        random_type=RandomType.PSEUDO_ANTITHETIC,
        seed=seed)

    sample = sample.asnumpy()
    print("test_mean_and_scale_antithetic")
    np.testing.assert_array_equal(sample.shape, (size, 2, 2))
    # Antithetic combination of samples should be equal to the `mean`
    antithetic_size = size // 2
    antithetic_combination = (sample[:antithetic_size, ...] +
                              sample[antithetic_size:, ...]) / 2
    np.testing.assert_allclose(antithetic_combination,
                               mean + np.zeros([antithetic_size, 2, 2]), 1e-6,
                               1e-6)
    # Get the antithetic pairs and verify normality
    np.testing.assert_array_almost_equal(np.mean(sample[:antithetic_size, ...],
                                                 axis=0),
                                         mean,
                                         decimal=1)
    np.testing.assert_array_almost_equal(np.cov(sample[:antithetic_size, 0, :],
                                                rowvar=False),
                                         covariance,
                                         decimal=1)
    np.testing.assert_array_almost_equal(np.cov(sample[:antithetic_size, 1, :],
                                                rowvar=False),
                                         covariance,
                                         decimal=1)


def test_default_multivariate_normal_sobol():
    """Tests that the default value of mean is 0."""
    covar = np.array([[1.0, 0.1], [0.1, 1.0]])
    skip = 1000
    sample = random_sampler.multivariate_normal((10000, ),
                                                covariance_matrix=covar,
                                                random_type=RandomType.SOBOL)
    sample = sample.asnumpy()

    np.testing.assert_equal(sample.shape, (10000, 2))
    print("test_default_multivariate_normal_sobol")
    print(np.mean(sample, axis=0))
    print(np.allclose(np.mean(sample, axis=0), [0.0, 0.0], 1e-3))
    print(
        np.isclose(
            np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]),
            2e-2))


def test_batch_multivariate_normal_sobol():
    """Tests sample for batch sobol."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0., 0.]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    covariance = np.matmul(scale, scale.transpose())
    sample_shape = (2, 3, 5)

    sample = random_sampler.multivariate_normal(sample_shape=sample_shape,
                                                mean=mean,
                                                scale_matrix=scale,
                                                random_type=RandomType.SOBOL)
    sample = sample.asnumpy()
    print("test_batch_multivariate_normal_sobol")
    np.testing.assert_array_equal(sample.shape, sample_shape + (4, 2))
    np.testing.assert_array_almost_equal(np.mean(sample, axis=(0, 1, 2)),
                                         mean,
                                         decimal=1)
    # result is different from tff
    # for i in range(4):
    #     np.testing.assert_array_almost_equal(np.cov(sample[0, 1, :, i, :],
    #                                                 rowvar=False),
    #                                          covariance,
    #                                          decimal=1)

    # print(f"sample: {sample}")


def test_mean_default_multivariate_normal_halton():
    """Tests that the default halton value of mean is 0."""
    mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0., 0.]])
    scale = np.array([[0.4, -0.1], [0.22, 1.38]])
    covariance = np.matmul(scale, scale.transpose())
    sample_shape = (2, 3, 5000)
    sample = random_sampler.multivariate_normal(sample_shape,
                                                mean=mean,
                                                scale_matrix=scale,
                                                random_type=RandomType.HALTON)
    sample = sample.asnumpy()
    print("test_mean_default_multivariate_normal_halton")
    np.testing.assert_array_equal(sample.shape, sample_shape + (4, 2))
    np.testing.assert_array_almost_equal(np.mean(sample, axis=(0, 1, 2)),
                                         mean,
                                         decimal=1)
    for i in range(4):
        np.testing.assert_array_almost_equal(np.cov(sample[0, 2, :, i, :],
                                                    rowvar=False),
                                             covariance,
                                             decimal=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random test")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        help="set which type of device you want to use. Ascend/GPU",
    )
    parser.add_argument("--device_id",
                        default=0,
                        type=int,
                        help="device id is for physical devices")
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
    # test_normal()

    # start = time.time()
    # test_normal()
    # end = time.time()
    # print(f"normal time {end - start}")

    # test_sobol()

    # start = time.time()
    # test_sobol()
    # end = time.time()
    # print(f"sobol time {end - start}")

    test_general_mean_covariance_pseudo()

    test_mean_and_scale_pseudo_antithetic()

    # test_batch_multivariate_normal_sobol()

    # test_default_multivariate_normal_sobol()

    # test_mean_default_multivariate_normal_halton()

    # test_halton_numbers_generation()
