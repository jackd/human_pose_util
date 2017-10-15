from __future__ import division
import unittest
import numpy as np

from human_pose_util.transforms.camera_params import \
    calculate_extrinsics
from human_pose_util.transforms.camera_params import \
    calculate_intrinsics_1d, calculate_intrinsics


class TestCameraIntrinsics(unittest.TestCase):
    """Test calculate_camera_intinsics functions."""

    def test_1d(self):
        n = 1000
        x3 = np.random.uniform(size=(n,))
        z3 = np.random.uniform(size=(n,)) + 10
        f = 2.0
        c = 3.4
        x2 = x3 / z3 * f + c
        actual_f, actual_c = calculate_intrinsics_1d(x3, z3, x2)
        np.testing.assert_allclose(actual_f, f)
        np.testing.assert_allclose(actual_c, c)

    def test_2d(self):
        n = 1000
        p3 = np.random.uniform(size=(n, 3))
        p3[:, 2] += 2
        f = np.array([1.2, 1.1])
        c = np.array([200., 201.])
        p2 = p3[:, :2] / p3[:, -1:] * f + c
        actual_f, actual_c = calculate_intrinsics(p3, p2)
        np.testing.assert_allclose(actual_f, f)
        np.testing.assert_allclose(actual_c, c)


class TestCameraExtrinsics(unittest.TestCase):
    """Test calculate_extrinsics."""

    def test_camera_extrinsics(self):
        from human_pose_util.transforms.np_impl import euler_matrix_nh
        from human_pose_util.transforms.np_impl import euler_from_matrix_nh
        # from transformations import euler_from_matrix
        n = 1000
        A = np.random.uniform(size=(n, 3))
        r = np.random.uniform(size=(3,))
        t = np.random.uniform(-1, 1, size=(3,))
        R = euler_matrix_nh(r[0], r[1], r[2])
        k = 1.3
        B = k * np.dot(A, R.T) + t
        actual_R, actual_t, actual_k = calculate_extrinsics(A, B)
        np.testing.assert_allclose(actual_R, R)
        np.testing.assert_allclose(actual_t, t)
        np.testing.assert_allclose(actual_k, k)
        actual_r = euler_from_matrix_nh(actual_R)
        np.testing.assert_allclose(actual_r, r)


if __name__ == '__main__':
    unittest.main()
