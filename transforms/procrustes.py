from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .np_impl import np_impl
import numpy as np
# import scipy.spatial
# scipy.spatial.procrustes


def orthogonal_procrustes(A, B, impl=np_impl):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.

    See `scipy.spatial.orthogonal_procrustes`

    Args:
        A: (B, M, N) array-like
        B: (B, M, N) array-like
        impl: implementation, e.g. tf_impl, np_impl

    Returns:
        R: (B, N, N) tensor - transform for each element of the batch
        scale: (B, 1) tensor - sum of singular vlaues of A.T @ B.
    """
    if impl.ndims(A) < 3:
        raise ValueError(
            'expected ndim to be 3, but observed %s' % impl.ndims(A))
    # if A.shape != B.shape:
    #     raise ValueError('the shapes of A and B differ (%s vs %s)' % (
    #         A.shape, B.shape))
    w, u, v = impl.svd(impl.batch_matmul(A, B, transpose_a=True))
    R = impl.batch_matmul(u, v, transpose_b=True)
    scale = impl.reduce_sum(w, axis=-1, keepdims=True)
    return R, scale


def procrustes(data1, data2, impl=np_impl, dtype=np.float32):
    """
    Procrustes analysis, a similarity test for two data sets.

    Generalization of `scipy.spatial.procrustes` for np/tf impl.

    First dimension of data1 and data2 is a batch dimension.

    Args:
        data1 : (B, N, K) tensor
        data2 : (B, N, K) tensor to transform to match `data1`
        impl : implementation - e.g. tf_impl, np_impl

    Returns:
    mtx1 : (B, N, K) tensor - standardized version of `data1`.
    mtx2 : (B, N, K) tensor - re-oriented `data2` to best match `data1`.
        Centered, but not necessarily `tr(AA^{T}) = 1`.
    norm1: (B, 1, 1) tensor - per-element frobenius norm of `data1`
    norm2: (B, 1, 1) tensor - per-element frobenius norm of `data2`
    offset1: (B, 1, 3) tensor - pre-element offset applied to `data1`
    offset2: (B, 1, 3) tensor - per-element offset applied to `data2`

    Raises
    ------
    ValueError
        If the input arrays are not three-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.
    """
    mtx1 = impl.array(data1, dtype=dtype, copy=True)
    mtx2 = impl.array(data2, dtype=dtype, copy=True)

    if impl.ndims(mtx1) < 3 or impl.ndims(mtx2) < 3:
        raise ValueError("Input matrices must be three-dimensional")
    # if mtx1.shape != mtx2.shape:
    #     raise ValueError("Input matrices must be of same shape")
    # if impl.num_elements(mtx1) == 0:
    #     raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    offset1 = impl.reduce_mean(mtx1, axis=-2, keepdims=True)
    offset2 = impl.reduce_mean(mtx2, axis=-2, keepdims=True)
    mtx1 -= offset1
    mtx2 -= offset2

    norm1 = impl.norm(mtx1, axis=(-2, -1), keepdims=True)
    norm2 = impl.norm(mtx2, axis=(-2, -1), keepdims=True)

    # if norm1 == 0 or norm2 == 0:
    #     raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2, impl)
    s = impl.expand_dims(s, axis=-1)
    mtx2 = impl.batch_matmul(mtx2, R, transpose_b=True) * s

    return mtx1, mtx2, norm1, norm2, offset1, offset2


def procrustes_aligned(data1, data2, impl=np_impl, dtype=np.float32):
    """
    Apply an optimal transformation to `data2` to align with `data1`.

    Args:
        See `procrustes`
    Returns:
        Transformed `data2` according to the rigid body transformation that
            minimizes the 2-norm with `data1`.
    """
    mtx1, mtx2, norm1, norm2, offset1, offset2 = procrustes(
        data1, data2, impl=impl, dtype=dtype)
    return mtx2 * norm1 + offset1
