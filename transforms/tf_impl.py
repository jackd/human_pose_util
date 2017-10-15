import tensorflow as tf
import numpy as np
from base import Transform


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

    def reduce_sum(self, x, axis=None, keep_dims=False):
        return tf.reduce_sum(x, axis=axis, keep_dims=keep_dims)

    def transpose(self, tensor, perm):
        return tf.transpose(tensor, perm)


tf_impl = _TransformTf()
batch_matmul = tf_impl.batch_matmul
rotation_matrix = tf_impl.rotation_matrix
rotation_with_matrix = tf_impl.rotation_with_matrix
rotate = tf_impl.rotate
rotate_about = tf_impl.rotate_about
transform_frame = tf_impl.transform_frame
project = tf_impl.project
atan2 = tf_impl.atan2
