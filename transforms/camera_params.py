from __future__ import division
import numpy as np


def calculate_intrinsics_1d(x3, z3, x2):
    """
    Calculate f and c values such that f3/x3 * f + c = x2.

    Args:
        x3: array length n, representing '3d' x coordinate.
        z3: array length n, representing '3d' z coordinate.
        x2: array length n, representing '2d' x coordinate.

    Returns:
        f: focal length in 1d
        c: pixel offset in 1d
    """
    n = x3.shape[0]
    assert(x3.shape == (n,))
    assert(x3.shape == z3.shape == (n,))
    assert(x2.shape == (n,))
    A = np.ones((n, 2))
    A[:, 0] = x3 / z3
    sol, res, rank, s = np.linalg.lstsq(A, x2)

    assert(sol.shape == (2,))
    f, c = sol
    return f, c


def calculate_intrinsics_2d(p3, p2, dtype=np.float32):
    """
    Calculate f and c values such that transforms.project(p3, f, c) = p2.

    i.e.
    p3[:, :2] / p3[:, 2:] * f + c = p2

    Args:
        p3: 3d coordinates, shape (n, 3)
        p2: 2d coordintaes, shape (n, 2)
    Returns:
        f: np array with 2 entries, focal lengths
        c: np array with 2 entries, pixel offset
    """
    n = p3.shape[0]
    assert(p3.shape == (n, 3))
    assert(p2.shape == (n, 2))
    fcs = [calculate_intrinsics_1d(
        p3[:, i], p3[:, 2], p2[:, i]) for i in range(2)]
    f = np.array([fc[0] for fc in fcs], dtype=dtype)
    c = np.array([fc[1] for fc in fcs], dtype=dtype)
    return f, c


calculate_intrinsics = calculate_intrinsics_2d


def calculate_extrinsics(A, B, overwrite=False):
    """
    Calculate rotation, translation and scale such that k*A*R.T + t = B.

    Args:
        A: matrix, shape (n, m)
        B: matrix, shape (n, m)
        overwrite: if true, values of A and B are overwritten

    Returns:
        R: matrix, shape (m, m). Orthogonal, i.e. dot(R, R.T) = 0
        t: vector, shape (n,)
        k: scalar
    such that k*np.dot(A, R.T) + t = B

    Base on scipy.spatial.procruste.
    """
    from scipy.linalg import orthogonal_procrustes
    if not overwrite:
        A = A.copy()
        B = B.copy()
    ta = np.mean(A, axis=0)
    tb = np.mean(B, axis=0)
    A -= ta
    B -= tb
    ka = np.linalg.norm(A)
    kb = np.linalg.norm(B)
    if ka == 0 or kb == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    A /= ka
    B /= kb
    R, S = orthogonal_procrustes(A, B)
    k = kb / ka
    t = tb - k*np.dot(ta, R)
    R = R.T
    return R, t, k


if __name__ == '__main__':
    from np_impl import project
    n = 100
    p3 = np.abs(np.random.random((n, 3)))
    p3[:, 2] += 10
    f = np.array([105, 101])
    c = np.array([1000, 2000])

    p2 = project(p3, f, c)

    fi, ci = calculate_intrinsics(p3, p2)
    print(fi - f)
    print(ci - c)
