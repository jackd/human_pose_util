from __future__ import division
import unittest
import numpy as np
import tensorflow as tf
import human_pose_util.transforms


class TestNPTransforms(unittest.TestCase):
    """Test numpy transformations."""
    @property
    def impl(self):
        return human_pose_util.transforms.np_impl

    def assertArraysEqual(self, actual, expected):
        np.testing.assert_equal(actual, expected)

    def assertArraysClose(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_stack(self):
        actual = self.impl.stack([1, 2, 3], axis=0)
        expected = np.array([1, 2, 3])
        self.assertArraysEqual(actual, expected)
        actual = self.impl.stack(
            [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])], axis=1)
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        self.assertArraysEqual(actual, expected)

    def test_unstack(self):
        actual = self.impl.unstack(np.array([1, 2, 3]))
        expected = [1, 2, 3]
        for a, e in zip(actual, expected):
            self.assertEqual(a, e)
        actual = self.impl.unstack(np.array([[1, 3, 5], [2, 4, 6]]), axis=1)
        expected = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        for a, e in zip(actual, expected):
            self.assertArraysEqual(a, e)

    def test_squeeze(self):
        actual = self.impl.squeeze([[1, 2, 3]], axis=0)
        expected = np.array([1, 2, 3])
        self.assertArraysEqual(actual, expected)
        actual = self.impl.squeeze(np.array([[1], [2], [3]]), axis=1)
        self.assertArraysEqual(actual, expected)

    def test_expand_dims(self):
        actual = self.impl.expand_dims(np.array([1, 2, 3]), axis=0)
        expected = np.array([[1, 2, 3]])
        self.assertArraysEqual(actual, expected)
        actual = self.impl.expand_dims(np.array([1, 2, 3]), axis=1)
        expected = np.array([[1], [2], [3]])
        self.assertArraysEqual(actual, expected)

    def test_reshape(self):
        actual = self.impl.reshape(np.array([1, 2, 3, 4]), (2, 2))
        expected = np.array([[1, 2], [3, 4]])
        self.assertArraysEqual(actual, expected)

    def test_split(self):
        actual = self.impl.split(
            np.array([[2, 3, 4], [5, 6, 7]]), [2, 1], axis=-1)
        expected = [np.array([[2, 3], [5, 6]]), np.array([[4], [7]])]
        for a, e in zip(actual, expected):
            self.assertArraysEqual(a, e)

    def test_matmul(self):
        A = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        B = np.array([[5, 6], [7, 8]])
        actual = self.impl.matmul(A, b)
        expected = np.array([17, 39])
        self.assertArraysEqual(actual, expected)
        actual = self.impl.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        self.assertArraysEqual(actual, expected)
        self.assertArraysEqual(
            self.impl.matmul(A.T, B, transpose_a=True), expected)
        self.assertArraysEqual(
            self.impl.matmul(A, B.T, transpose_b=True), expected)

    def test_batch_matmul(self):
        A = np.expand_dims(np.array([[1, 2], [3, 4]]), axis=0)
        B = np.expand_dims(np.array([[5, 6], [7, 8]]), axis=0)
        actual = self.impl.batch_matmul(A, B)
        expected = np.expand_dims(np.array([[19, 22], [43, 50]]), axis=0)
        self.assertArraysEqual(actual, expected)
        A = np.array([A[0], A[0]], dtype=np.float32)
        B = np.array([B[0], B[0]], dtype=np.float32)
        expected = np.array([expected[0], expected[0]], dtype=np.float32)
        self.assertArraysEqual(self.impl.batch_matmul(A, B), expected)
        At = self.impl.transpose(A, [0, 2, 1])
        Bt = self.impl.transpose(B, [0, 2, 1])
        self.assertArraysEqual(
            self.impl.batch_matmul(At, B, transpose_a=True), expected)
        self.assertArraysEqual(
            self.impl.batch_matmul(A, Bt, transpose_b=True), expected)
        self.assertArraysEqual(
            self.impl.batch_matmul(At, Bt, transpose_a=True, transpose_b=True),
            expected)
        batch_size = 10
        A = np.random.random((batch_size, 4, 4)).astype(np.float32)
        B = np.random.random((batch_size, 4, 4)).astype(np.float32)

        def _test(A, B, transpose_a=False, transpose_b=False):
            actual = self.impl.batch_matmul(A, B, transpose_a, transpose_b)
            expected = np.array([np.matmul(
                a.T if transpose_a else a, b.T if transpose_b else b)
                for a, b in zip(A, B)], dtype=np.float32)
            self.assertArraysClose(actual, expected)

        for transpose_a in [False, True]:
            for transpose_b in [False, True]:
                _test(A, B, transpose_a, transpose_b)
        expected = self.impl.batch_matmul(A, B)
        A = np.array([A, A])
        # B = np.expand_dims(B, axis=0)
        B = np.array([B, B])
        actual = self.impl.batch_matmul(A, B)
        expected = self.impl.stack([expected, expected], axis=0)
        self.assertArraysClose(actual, expected)

    def test_general_matmul(self):
        n = 2
        m = 3
        A = np.random.random((n, 4, 4))
        B = np.random.random((4, 4, m))

        def _test(A, B, transpose_a=False, transpose_b=False):
            actual = self.impl.general_matmul(A, B, transpose_a, transpose_b)
            expected = np.empty((n, 4, 4, m))
            for i in range(n):
                a = A[i].T if transpose_a else A[i]
                for j in range(m):
                    b = B[..., j].T if transpose_b else B[..., j]
                    expected[i, ..., j] = np.matmul(a, b)
            self.assertArraysClose(actual, expected)

        p = [True, False]
        for transpose_a in p:
            for transpose_b in p:
                _test(A, B, transpose_a, transpose_b)

    def test_rotation_matrix(self):
        r = np.array([0, 0, np.pi/3])
        c = 1/2
        s = np.sqrt(3)/2
        actual = self.impl.rotation_matrix(r)
        expected = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        self.assertArraysClose(actual, expected)
        r = np.array([[0, 0, np.pi/3], [0, 0, np.pi/6]])
        actual = self.impl.rotation_matrix(r, stack_axis=-1)
        expected = np.array([
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ],
            [
                [s, -c, 0],
                [c, s, 0],
                [0, 0, 1]
            ]
        ])
        self.assertArraysClose(actual, expected)
        actual = self.impl.rotation_matrix(r, stack_axis=0)
        expected = np.transpose(expected, (1, 2, 0))
        self.assertArraysClose(actual, expected)

    def test_rotate(self):
        points = np.array([[1, 0, 0]], dtype=np.float32)
        r = np.array([0, 0, np.pi/3], dtype=np.float32)
        expected = np.array([[1/2, np.sqrt(3)/2, 0]], dtype=np.float32)
        actual = self.impl.rotate(points, r, inverse=False)
        self.assertArraysClose(actual, expected)
        # inverse
        points = np.random.random((24, 3)).astype(np.float32)
        r = np.random.random((3,)).astype(np.float32)
        self.assertArraysClose(self.impl.rotate(
            self.impl.rotate(points, r), r, inverse=True), points)

    def test_rotate_about(self):
        points = np.array([[1, 0, 0]], dtype=np.float32)
        angle = np.array(np.pi/3, dtype=np.float32)
        expected = np.array([[1/2, np.sqrt(3)/2, 0]], dtype=np.float32)
        actual = self.impl.rotate_about(points, angle, 2)
        self.assertArraysClose(actual, expected)

    def test_transform_frame(self):
        points = np.array([[1, 0, 0]], dtype=np.float32)
        r = np.array([0, 0, np.pi/3], dtype=np.float32)
        t = np.array([2, 3, 4], dtype=np.float32)
        actual = self.impl.transform_frame(points, r, t)
        expected = np.array(
            [[2 + 1/2, 3 + np.sqrt(3)/2, 4 + 0]], dtype=np.float32)
        self.assertArraysClose(actual, expected)
        # inverse
        batch_size = 15
        points = np.random.random((batch_size, 24, 3)).astype(np.float32)
        r = np.random.random((batch_size, 3)).astype(np.float32)
        t = np.random.random((batch_size, 3)).astype(np.float32)
        self.assertArraysClose(
            self.impl.transform_frame(
                self.impl.transform_frame(points, r, t), r, t, inverse=True),
            points)

    def test_project(self):
        points = np.array([[1, 2, 1], [2, 2, 2]], dtype=np.float32)
        f = np.array(1., dtype=np.float32)
        c = np.array(0., dtype=np.float32)
        actual = self.impl.project(points, f, c)
        expected = np.array([[1, 2], [1, 1]], dtype=np.float32)
        self.assertArraysClose(actual, expected)


class TestTFTransforms(TestNPTransforms):
    """Test tensorflow transformations."""
    @property
    def impl(self):
        return human_pose_util.transforms.tf_impl

    def setUp(self):
        tf.reset_default_graph()

    def _to_numpy(self, arr, sess):
        if isinstance(arr, tf.Tensor):
            arr = sess.run(arr)
        return arr

    def assertEqual(self, actual, expected):
        with tf.Session() as sess:
            actual = self._to_numpy(actual, sess)
            expected = self._to_numpy(expected, sess)
        super(TestTFTransforms, self).assertEqual(actual, expected)

    def assertArraysEqual(self, actual, expected):
        with tf.Session() as sess:
            actual = self._to_numpy(actual, sess)
            expected = self._to_numpy(expected, sess)
        super(TestTFTransforms, self).assertArraysEqual(actual, expected)

    def assertArraysClose(self, actual, expected):
        with tf.Session() as sess:
            actual = self._to_numpy(actual, sess)
            expected = self._to_numpy(expected, sess)
        super(TestTFTransforms, self).assertArraysClose(actual, expected)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.logging.set_verbosity(0)
    unittest.main()
