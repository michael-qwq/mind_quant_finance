import pytest

from mindspore import dtype as mstype

from mind_quant_finance.engine.mc.mc_brownian_motion import McBrownianModel
from mind_quant_finance.engine.mc.mc_utils import *
from mind_quant_finance.math.random_ops.random import RandomType

dtype = mstype.float32
NUM_MC = 100000
NUM_STDERRS = 3.0  # Maximum size of the error in multiples of the standard error


class TestMcBrownianModel:

    def test_default_initial(self):
        """Tests that the default & initial are correct."""
        mean = ms.Tensor([[0.5], [0.5]])
        volatility = ms.Tensor([[0.5], [0.5]])
        times = ms.Tensor([1., 2., 3., 4.])
        ito_process = McBrownianModel(mean, volatility)
        samples = ito_process(times)
        assert samples.ndim == 4
        assert samples.shape[0] == 2
        assert samples.shape[1] == 1
        assert samples.shape[2] == 5
        assert samples.shape[3] == 1
        assert (np.abs(samples[:, :, 0, 0] - np.ones((2, 1))) < 1e-6).all()

    def test_univariate_sample_mean_and_variance_constant_parameters(self):
        """Tests the mean and vol of the univariate GBM sampled paths."""
        mu = 0.05
        sigma = 0.05
        times = np.array([0.1, 0.5, 1.0], dtype=dtype)
        initial_state = 2.0
        ito_process = McBrownianModel(mu, sigma)
        rtype = RandomType.PSEUDO.value
        mean, var, se_mean, se_var = \
            calculate_mean_and_variance_from_logdelta_sample_paths(
                ito_process(times, initial_state, NUM_MC, rtype, 0))
        mu, sigma, initial_state = ms.Tensor([[[mu]]]), ms.Tensor([[[sigma]]]), ms.Tensor([[[initial_state]]])
        times = np.expand_dims(times, -1)
        expected_mean = (mu - sigma ** 2 / 2) * times
        expected_var = sigma ** 2 * times
        assert (np.abs(mean - expected_mean) < se_mean * NUM_STDERRS).all()
        assert (np.abs(var - expected_var) < se_var * NUM_STDERRS).all()

    def test_univariate_sample_mean_constant_parameters_batched(self):
        """Tests the mean and vol of the batched univariate GBM sampled paths."""
        # Batch dimensions [4].
        mu = np.array([[0.05], [0.06], [0.04], [0.02]], dtype=dtype)
        sigma = np.array([[0.05], [0.1], [0.15], [0.2]], dtype=dtype)
        times = np.array([0.1, 0.5, 1.0], dtype=dtype)
        initial_state = np.array([[2.0], [10.0], [5.0], [25.0]], dtype=dtype)
        ito_process = McBrownianModel(mu, sigma)
        rtype = RandomType.PSEUDO.value
        mean, var, se_mean, se_var = \
            calculate_mean_and_variance_from_logdelta_sample_paths(
                ito_process(times, initial_state, NUM_MC, rtype, 0))
        times = np.expand_dims(times, -1)  # (n_times, 1)
        mu = np.expand_dims(mu, -2)
        sigma = np.expand_dims(sigma, -2)  # (batch, 1, 1)
        expected_mean = (mu - sigma ** 2 / 2) * times
        expected_var = sigma ** 2 * times

        assert (np.abs(mean - expected_mean) < se_mean * NUM_STDERRS).all()
        assert (np.abs(var - expected_var) < se_var * NUM_STDERRS).all()

    def test_univariate_sample_mean_constant_parameters_batched_time(self):
        """Tests the mean and vol of the batched univariate GBM sampled paths."""
        # Batch dimensions [4].
        mu = np.array([[0.05], [0.06], [0.04], [0.03]], dtype=dtype)
        sigma = np.array([[0.05], [0.1], [0.15], [0.2]], dtype=dtype)
        times = np.array([[0.1, 0.5, 1.0],
                          [0.2, 0.4, 2.0],
                          [0.3, 0.6, 5.0],
                          [0.4, 0.9, 7.0]], dtype=dtype)
        initial_state = np.array([[2.0], [10.0], [5.0], [25.0]], dtype=dtype)
        ito_process = McBrownianModel(mu, sigma)
        rtype = RandomType.PSEUDO.value
        mean, var, se_mean, se_var = \
            calculate_mean_and_variance_from_logdelta_sample_paths(
                ito_process(times, initial_state, NUM_MC, rtype, 0))
        times = np.expand_dims(times, -1)  # (batch, times, 1)
        mu = np.expand_dims(mu, -2)
        sigma = np.expand_dims(sigma, -2)  # (batch, 1, 1)
        expected_mean = (mu - sigma ** 2 / 2) * times
        expected_var = sigma ** 2 * times

        assert (np.abs(mean - expected_mean) < se_mean * NUM_STDERRS).all()
        assert (np.abs(var - expected_var) < se_var * NUM_STDERRS).all()

    def test_univariate_sample_mean_constant_parameters_batched2(self):
        """Tests the mean and vol of the batched univariate GBM sampled paths."""
        # Batch dimensions [4].
        mu = np.array([[0.05, 0.06, 0.04, 0.02]], dtype=dtype)
        sigma = np.array([[0.05, 0.1, 0.15, 0.2]], dtype=dtype)
        times = np.array([0.1, 0.5, 1.0], dtype=dtype)
        initial_state = np.array([[2.0, 10.0, 5.0, 25.0]], dtype=dtype)
        ito_process = McBrownianModel(mu, sigma)
        rtype = RandomType.PSEUDO.value
        mean, var, se_mean, se_var = \
            calculate_mean_and_variance_from_logdelta_sample_paths(
                ito_process(times, initial_state, NUM_MC, rtype, 0))
        times = np.expand_dims(times, -1)  # (n_times, 1)
        mu = np.expand_dims(mu, -2)
        sigma = np.expand_dims(sigma, -2)  # (batch, 1, 4)
        expected_mean = (mu - sigma ** 2 / 2) * times
        expected_var = sigma ** 2 * times

        assert (np.abs(mean - expected_mean) < se_mean * NUM_STDERRS).all()
        assert (np.abs(var - expected_var) < se_var * NUM_STDERRS).all()

    def test_univariate_sample_mean_constant_parameters_batched2_time(self):
        """Tests the mean and vol of the batched univariate GBM sampled paths."""
        # Batch dimensions [4].
        mu = np.array([[0.05, 0.06, 0.04, 0.02]], dtype=dtype)
        sigma = np.array([[0.05, 0.1, 0.15, 0.2]], dtype=dtype)
        times = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 7.0]], dtype=dtype)
        initial_state = np.array([[2.0, 10.0, 5.0, 25.0]], dtype=dtype)
        ito_process = McBrownianModel(mu, sigma)
        rtype = RandomType.PSEUDO.value
        mean, var, se_mean, se_var = \
            calculate_mean_and_variance_from_logdelta_sample_paths(
                ito_process(times, initial_state, NUM_MC, rtype, 0))
        times = np.expand_dims(times, -1)  # (batch, times, 1)
        mu = np.expand_dims(mu, -2)
        sigma = np.expand_dims(sigma, -2)  # (batch, 1, 1)
        expected_mean = (mu - sigma ** 2 / 2) * times
        expected_var = sigma ** 2 * times

        assert (np.abs(mean - expected_mean) < se_mean * NUM_STDERRS).all()
        assert (np.abs(var - expected_var) < se_var * NUM_STDERRS).all()

    def test_multivariate_sample_mean_and_variance_nocorr(self):
        """Tests the mean and vol of the univariate GBM sampled paths."""
        means = 0.05
        dim = 2
        volatilities = np.array([[0.1, 0.2]])
        times = np.array([0.1, 0.5, 1.0])
        initial_state = np.array([[1.0, 2.0]])
        num_samples_local = 100000
        ito_process = McBrownianModel(means, volatilities)
        rtype = RandomType.PSEUDO.value
        samples = ito_process(times, initial_state, num_samples_local, rtype, 0)
        mean, var, se_mean, se_var = calculate_mean_and_variance_from_logdelta_sample_paths(samples)
        means = ms.Tensor([[means]])
        times = np.expand_dims(times, -1)  # (batch, times, 1)
        means = np.expand_dims(means, -2)
        volatilities = np.expand_dims(volatilities, -2)  # (batch, 1, 2)
        expected_mean = ((means - volatilities ** 2 / 2) * times)
        expected_var = volatilities ** 2 * times
        n_time = times.shape[1]

        for b in range(samples.shape[0]):   # (batch, num_path, times, dim)
            for i in range(n_time):
                corr = np.corrcoef(samples[b, :, i+1, :].T)  # (num_sample, dim)
                for j1 in range(dim):
                    for j2 in range(j1+1, dim):
                        assert (np.abs(corr[j1, j2]) < 1e-1).all()
        assert (np.abs(mean - expected_mean) < 1e-2).all()
        assert (np.abs(var - expected_var) < 1e-2).all()

    def test_multivariate_sample_mean_and_variance_corr(self):
        """Tests the mean and vol of the univariate GBM sampled paths."""
        dim = 2
        means = 0.05
        volatilities = np.array([[0.1, 0.2]])
        corr_matrix = np.array([[[1, 0.1], [0.1, 1]]])
        times = np.array([0.1, 0.5, 1.0])
        initial_state = np.array([[1.0, 2.0]])
        num_samples_local = 100000
        ito_process = McBrownianModel(means, volatilities, corr_matrix)
        rtype = RandomType.PSEUDO.value
        samples = ito_process(times, initial_state, num_samples_local, rtype, 0)
        mean, var, se_mean, se_var = calculate_mean_and_variance_from_logdelta_sample_paths(samples)
        means = ms.Tensor([[means]])
        times = np.expand_dims(times, -1)  # (batch, times, 1)
        means = np.expand_dims(means, -2)
        volatilities = np.expand_dims(volatilities, -2)  # (batch, 1, 2)
        expected_mean = ((means - volatilities ** 2 / 2) * times)
        expected_var = volatilities ** 2 * times
        n_time = times.shape[1]

        for b in range(samples.shape[0]):
            for i in range(n_time):
                corr = np.corrcoef(samples[b, :, i + 1, :].T)  # (num_sample, dim)
                for j1 in range(dim):
                    for j2 in range(j1 + 1, dim):
                        st = corr_matrix[b, j1, j2]
                        assert (np.abs(corr[j1, j2] - st) < np.abs(st)*0.1).all()
        assert (np.abs(mean - expected_mean) < 1e-2).all()
        assert (np.abs(var - expected_var) < 1e-2).all()
        assert (np.abs(corr_matrix - ito_process.corr_matrix()) < 1e-2).all()

    def test_multivariate_sample_mean_and_variance_corr_batched(self):
        """Tests the mean and vol of the univariate GBM sampled paths."""
        dim = 2
        means = np.array([[0.02, 0.05], [0.04, 0.01]])
        volatilities = np.array([[0.2, 0.1], [0.1, 0.15]])
        corr_matrix = np.array([[[1, 0.1], [0.1, 1]], [[1, 0.4], [0.4, 1]]])
        times = np.array([[0.1, 0.5, 1.0], [0.2, 0.8, 2.0]])
        initial_state = np.array([[1.0, 2.0], [2.0, 1.0]])
        num_samples_local = 100000
        ito_process = McBrownianModel(means, volatilities, corr_matrix)
        rtype = RandomType.PSEUDO.value
        samples = ito_process(times, initial_state, num_samples_local, rtype, 0)
        mean, var, se_mean, se_var = calculate_mean_and_variance_from_logdelta_sample_paths(samples)
        times = np.expand_dims(times, -1)  # (batch, times, 1)
        means = np.expand_dims(means, -2)
        volatilities = np.expand_dims(volatilities, -2)  # (batch, 1, 2)
        expected_mean = ((means - volatilities ** 2 / 2) * times)
        expected_var = volatilities ** 2 * times
        n_time = times.shape[1]

        for b in range(samples.shape[0]):
            for i in range(n_time):
                corr = np.corrcoef(samples[b, :, i + 1, :].T)  # (num_sample, dim)
                for j1 in range(dim):
                    for j2 in range(j1 + 1, dim):
                        st = corr_matrix[b, j1, j2]
                        assert (np.abs(corr[j1, j2] - st) < np.abs(st)*0.1).all()
        assert (np.abs(mean - expected_mean) < 1e-2).all()
        assert (np.abs(var - expected_var) < 1e-2).all()
        assert (np.abs(corr_matrix - ito_process.corr_matrix()) < np.abs(corr_matrix)*0.1).all()


if __name__ == '__main__':
    pytest.main('test_mc_brownian_model.py')


