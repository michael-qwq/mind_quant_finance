import numpy as np

import mindspore
from mindspore import Tensor
import mindspore.ops as P
import mindspore.numpy as mnp

# def prepare_grid(times,
#                  time_step,
#                  dtype,
#                  tolerance=None,
#                  num_time_steps=None,
#                  times_grid=None):
#     """Prepares grid of times for path generation.
#     """

#     if tolerance is None:
#         if dtype == mindspore.float64:
#             tolerance = 1e-6
#         elif dtype == mindspore.float32:
#             tolerance = 1e-3
#     tolerance = Tensor(tolerance, dtype)
#     if times_grid is None:
#         if num_time_steps is None:
#             all_times, time_indices = _grid_from_time_step(times=times,
#                                                            time_step=time_step,
#                                                            dtype=dtype,
#                                                            tolerance=tolerance)
#         else:
#             all_times, time_indices = _grid_from_num_times(
#                 times=times,
#                 time_step=time_step,
#                 num_time_steps=num_time_steps)
#     else:
#         all_times = times_grid
#         time_indices = times_grid.searchsorted(times_grid, times)
#         # Adjust indices to bring `times` closer to `times_grid`.
#         times_diff_1 = tf.gather(times_grid, time_indices) - times
#         times_diff_2 = tf.gather(times_grid,
#                                  tf.math.maximum(time_indices - 1, 0)) - times
#         time_indices = tf.where(
#             tf.math.abs(times_diff_2) > tf.math.abs(times_diff_1),
#             time_indices, tf.math.maximum(time_indices - 1, 0))


def _grid_from_time_step(*, times, time_step, dtype, tolerance):
    """Creates a time grid from an input time step."""
    # `grid` is genernated by `time_step`
    grid = mnp.arange(start=0.0, stop=times[-1], step=time_step, dtype=dtype)
    # concat original `times` and `grid`, some elements in `times` may be duplicate in `grid`
    times = Tensor(times, dtype)
    all_times = P.Concat(axis=0)([times, grid])
    (all_times, _) = P.Sort()(all_times)

    # if two float elements are duplicated, their difference is smaller than `tolerance`
    # just remove elements that the differences is smaller than `tolerance`
    # get the differences of the continuous two elements: all_times[i+1] - all_times[i]
    dt = all_times[1:] - all_times[:-1]
    dt = P.Concat(axis=-1)([Tensor([1.0], dtype), dt])
    duplicate_mask = P.Greater()(dt, tolerance)
    all_times = mnp.select(duplicate_mask, all_times)
    time_indices = all_times.searchsorted(times)
    time_indices = P.Minimum()(time_indices, all_times.shape[0] - 1)

    # Move `time_indices` to the left, if the requested `times` are removed from
    # `all_times` during deduplication
    time_indices = mnp.where(
        P.Gather()(all_times, time_indices) - times > tolerance,
        time_indices - 1, time_indices)

    return all_times, time_indices
