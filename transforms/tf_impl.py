from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .base import Transform


class _TransformTf(Transform):
    def eps(self, dtype):
        return np.finfo(dtype.as_numpy_dtype).eps * 4.0

    def ternary(self, condition, if_true_fn, if_false_fn):
        return tf.where(condition, if_true_fn(), if_false_fn())

    def sqrt(self, x):
        return tf.sqrt(x)

    def stack(self, values, axis):
        return tf.stack(values, axis=axis)

    def unstack(self, value, num=None, axis=0):
        return tf.unstack(value, num=num, axis=axis)

    def squeeze(self, input, axis=None):
        return tf.squeeze(input, axis=axis)

    def expand_dims(self, input, axis=None):
        return tf.expand_dims(input, axis=axis)

    def cos(self, x, name=None):
        return tf.cos(x, name=name)

    def sin(self, x, name=None):
        return tf.sin(x, name=name)

    def tan(self, x, name=None):
        return tf.tan(x, name=name)

    def atan2(self, y, x, name=None):
        return tf.atan2(y, x, name=name)

    def einsum(self, equation, *inputs, **kwargs):
        inputs = [tf.constant(inp, dtype=tf.float32)
                  if isinstance(inp, np.ndarray) else inp for inp in inputs]
        return tf.einsum(equation, *inputs, **kwargs)

    def reshape(self, tensor, shape, name=None):
        return tf.reshape(tensor, shape, name=name)

    def split(self, value, num_or_size_splits, axis=0, name='split'):
        return tf.split(value, num_or_size_splits, axis=axis, name=name)

    def reduce_sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

    def reduce_mean(self, x, axis=None, keepdims=False):
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

    def transpose(self, tensor, perm):
        return tf.transpose(tensor, perm)

    def svd(self, tensor, full_matrices=False, compute_uv=True):
        return tf.svd(tensor, full_matrices, compute_uv)

    def matrix_determinant(self, input):
        return tf.matrix_determinant(input)

    def sign(self, x):
        return tf.sign(x)

    def array(self, data, dtype=None, copy=True):
        if isinstance(data, (tf.Tensor, tf.Variable)):
            if data.dtype == dtype:
                return data
            else:
                return tf.cast(data, dtype)
        elif isinstance(data, (list, tuple, int, float, np.ndarray)):
            return tf.constant(data, dtype=dtype)
        else:
            raise TypeError('Unrecognized type for `data`: "%s"' % type(data))

    def ndims(self, x):
        return x.shape.ndims

    def num_elements(self, x):
        return x.shape.num_elements()

    def norm(self, x, ord=None, axis=None, keepdims=False):
        if ord is None:
            ord = 'fro'
        return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def square(self, x):
        return tf.square(x)


tf_impl = _TransformTf()
batch_matmul = tf_impl.batch_matmul
rotation_matrix = tf_impl.rotation_matrix
rotation_with_matrix = tf_impl.rotation_with_matrix
rotate = tf_impl.rotate
rotate_about = tf_impl.rotate_about
transform_frame = tf_impl.transform_frame
project = tf_impl.project
atan2 = tf_impl.atan2
