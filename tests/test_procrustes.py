#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from human_pose_util.transforms.np_impl import np_impl
from human_pose_util.transforms.tf_impl import tf_impl
from human_pose_util.transforms.procrustes import procrustes
import scipy.spatial as sp

B = 5
N = 10
M = 3
x = np.random.normal(size=(B, N, M))
y = np.random.normal(size=(B, N, M))

e1, e2, _ = zip(
    *(sp.procrustes(xi, yi) for xi, yi in zip(x, y)))
expected = e1, e2


class TestProcrustes(unittest.TestCase):
    def test_np_impl(self):
        actual = procrustes(x, y, impl=np_impl)[:2]
        for a, c in zip(actual, expected):
            np.testing.assert_allclose(a, c, atol=1e-6, rtol=1e-6)

    def test_tf_impl(self):
        actual = procrustes(x, y, impl=tf_impl)[:2]
        with tf.Session() as sess:
            actual = sess.run(actual)
        for a, c in zip(actual, expected):
            np.testing.assert_allclose(a, c, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # tf.logging.set_verbosity(0)
    unittest.main()
