from __future__ import division
import unittest
import numpy as np

from human_pose_util.transforms.np_impl import euler_matrix_nh
from human_pose_util.transforms.np_impl import euler_from_matrix_nh
from human_pose_util.transforms.np_impl import project
from human_pose_util.transforms.np_impl import transform_frame
from human_pose_util.transforms.camera_params import \
    calculate_extrinsics
from human_pose_util.transforms.camera_params import \
    calculate_intrinsics

from human_pose_util.dataset.h3m.consistent_example import get_train_examples
from human_pose_util.dataset.h3m.consistent_example import get_eval_examples


def _all_examples():
    return list(get_train_examples()) + list(get_eval_examples())


class TestH3mExamples(unittest.TestCase):
    """Test h3m example consistency."""
    def _test_example_intrinsics(self, example):
        p3c, p2, f, c = (example[k] for k in ['p3c', 'p2', 'f', 'c'])
        actual_f, actual_c = calculate_intrinsics(
            np.reshape(p3c, (-1, 3)), np.reshape(p2, (-1, 2)))
        np.testing.assert_allclose(actual_f, f)
        np.testing.assert_allclose(actual_c, c)

        actual_p2 = project(p3c, f, c)
        np.testing.assert_allclose(actual_p2, p2, atol=1.0)

    def test_intrinsics(self):
        examples = _all_examples()
        for example in examples:
            self._test_example_intrinsics(example)

    def _test_example_extrinsics(self, example):
        actual = {'k': 1.0}
        expected = {}
        p3w, p3c, actual['r'], actual['t'] = (
            example[k].astype(np.float32) for k in ['p3w', 'p3c', 'r', 't'])
        actual['R'] = euler_matrix_nh(*actual['r'])
        expected['R'], expected['t'], expected['k'] = \
            calculate_extrinsics(
                np.reshape(p3w, (-1, 3)), np.reshape(p3c, (-1, 3)))
        expected['r'] = np.array(euler_from_matrix_nh(expected['R']))
        assert(len(actual) == len(expected))
        for k in ['R', 'r', 't', 'k']:
            np.testing.assert_allclose(actual[k], expected[k], atol=1.0)

    def test_extrinsics(self):
        for example in _all_examples():
            self._test_example_extrinsics(example)

    def _test_example_transforms(self, example):
        p3w, p3c, r, t = (example[k] for k in ['p3w', 'p3c', 'r', 't'])
        actual_p3c = transform_frame(p3w, r, t)
        np.testing.assert_allclose(actual_p3c, p3c, atol=1.0)

    def test_transforms(self):
        for example in _all_examples():
            self._test_example_transforms(example)


if __name__ == '__main__':
    unittest.main()
