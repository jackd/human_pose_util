import numpy as np
from base import Transform


class _TransformNp(Transform):
    def eps(self, dtype):
        return np.finfo(float).eps * 4.0

    def ternary(self, condition, if_true_fn, if_false_fn):
        return if_true_fn() if condition else if_false_fn()

    def sqrt(self, x):
        return np.sqrt(x)

    def stack(self, values, axis):
        return np.stack(values, axis=axis)

    def unstack(self, value, num=None, axis=0):
        if num is None:
            num = value.shape[axis]
        return [np.squeeze(v, axis=axis)
                for v in np.split(value, num, axis=axis)]

    def squeeze(self, input, axis=None):
        return np.squeeze(input, axis=axis)

    def expand_dims(self, input, axis=None):
        return np.expand_dims(input, axis=axis)

    def cos(self, x):
        return np.cos(x)

    def sin(self, x):
        return np.sin(x)

    def tan(self, x, **kwargs):
        return np.tan(x)

    def atan2(self, y, x, **kwargs):
        return np.arctan2(y, x)

    def einsum(self, *args):
        return np.einsum(*args)

    def reshape(self, tensor, shape):
        return np.reshape(tensor, shape)

    def split(self, value, num_or_size_splits, axis=0):
        if not isinstance(num_or_size_splits, int):
            num_or_indices = np.cumsum(num_or_size_splits)
            assert(num_or_indices[-1] == value.shape[axis])
            num_or_indices = num_or_indices[:-1]
        else:
            num_or_indices = num_or_size_splits
        return np.split(value, num_or_indices, axis=axis)

    def reduce_sum(self, x, axis=None, keep_dims=False):
        return np.sum(x, axis=axis, keepdims=keep_dims)

    def matmul(self, A, B, transpose_a=False, transpose_b=False):
        return np.matmul(A.T if transpose_a else A, B.T if transpose_b else B)

    def transpose(self, tensor, perm):
        return np.transpose(tensor, perm)


np_impl = _TransformNp()
batch_matmul = np_impl.batch_matmul
rotation_matrix = np_impl.rotation_matrix
rotation_with_matrix = np_impl.rotation_with_matrix
euler_matrix_nh = np_impl.euler_matrix_nh
rotate = np_impl.rotate
transform_frame = np_impl.transform_frame
project = np_impl.project
atan2 = np_impl.atan2
euler_from_matrix_nh = np_impl.euler_from_matrix_nh
rotate_about = np_impl.rotate_about
