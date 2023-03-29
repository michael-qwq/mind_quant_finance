import mindspore
import mindspore.ops as P
from mindspore import Tensor
from mindspore import scipy as mscipy


def matvec(a: Tensor, b: Tensor):
    """Multiplies matrix `a` by vector `b`, producing `a` * `b`.
    `b` can also be a matrix.
    """
    return P.Squeeze(-1)(P.matmul(a, P.ExpandDims()(b, -1)))


def cholesky(matrix: Tensor):
    r"""
    Batch Cholesky Decomposition.

    Input:
        - **matrix** (Tensor) - A Tensor with shape = (batch, dim, dim), matrix to be decomposited.

    Output:
        - **cholesky_matrix** (Tensor) - A Tensor with shape = (batch, dim, dim),
                             while y[i, :, :] is the cholesky decomposition of x[i, :, :].
    """
    return P.vmap(mscipy.linalg.cholesky, in_axes=(0,), out_axes=0)(matrix)