from mindspore import ms_function
import mindspore.numpy as np
import mindspore as ms


@ms_function
def calculate_mean_and_variance_from_sample_paths(samples):
    r"""
    Returns the mean and variance of sample_paths

    Inputs:
        - **samples** (Tensor) - Sample paths with shape=(batch, num_sample, times, dim).

    Outputs:
        - **mean** (Tensor) - The mean on the num_sample dimension, Tensor with shape=(batch, times, dim).
        - **var** (Tensor) - The variance on the num_sample dimension, Tensor with shape=(batch, times, dim).
        - **std_err_mena** (Tensor) - Standard error of mean.
                                      See '<https://en.wikipedia.org/wiki/Standard_error>' for more detailed.
        - **std_err_var** (Tensor) - Standard error of variance.
                                     See Ahn, Fessler et al. for more detailed.
                                     '<https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf>'
    """
    num_sample = samples.shape[1]
    mean = np.mean(samples, axis=1, keepdims=True)
    var = np.mean((samples - mean) ** 2, axis=1)
    mean = np.squeeze(mean, axis=1)
    std_err_mean = np.sqrt(var / num_sample)
    std_err_var = var * np.sqrt(ms.Tensor(2.0 / (num_sample - 1.0)))
    return mean, var, std_err_mean, std_err_var


@ms_function
def calculate_mean_and_variance_from_logdelta_sample_paths(samples):
    r"""
    Returns the mean and variance of logdelta(sample_paths).
    Dpath = log(path)[1:] - log(path)[0], then Returns the mean and variance of Dpath.

    Inputs:
        - **samples** (Tensor) - Sample paths with shape=(batch, num_sample, times, dim).

    Outputs:
        - **mean** (Tensor) - The mean on the num_sample dimension, Tensor with shape=(batch, times-1, dim).
        - **var** (Tensor) - The variance on the num_sample dimension, Tensor with shape=(batch, times-1, dim).
        - **std_err_mena** (Tensor) - Standard error of mean.
                                      See '<https://en.wikipedia.org/wiki/Standard_error>' for more detailed.
        - **std_err_var** (Tensor) - Standard error of variance.
                                     See Ahn, Fessler et al. for more detailed.
                                     '<https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf>'
    """
    samples = np.log(samples)
    samples = samples[:, :, 1:] - samples[:, :, 0:1]
    return calculate_mean_and_variance_from_sample_paths(samples)