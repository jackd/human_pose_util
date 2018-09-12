from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _put_at(inds, axis=-1, slc=(slice(None),)):
    return (axis < 0)*(Ellipsis,) + axis*slc + (inds,) + (-1-axis)*slc


def put(x, inds, values, axis=0):
    """Assignment equivalent to np.take."""
    x[_put_at(inds, axis=axis)] = values


def get_transform_indices(input_skeleton, output_skeleton, oi_map={}):
    indices = []
    for i in range(output_skeleton.n_joints):
        joint = output_skeleton.joint(i)
        joint = oi_map.get(joint, joint)
        indices.append(input_skeleton.joint_index(joint))
    return indices


class SkeletonConverter(object):
    def __init__(self, input_skeleton, output_skeleton, oi_map={}):
        self._indices = tuple(get_transform_indices(
            input_skeleton, output_skeleton, oi_map))
        self.input_skeleton = input_skeleton
        self.output_skeleton = output_skeleton

    @property
    def indices(self):
        return self._indices

    def convert(self, input_data, axis=-2):
        return np.take(input_data, self._indices, axis=axis)

    def convert_tf(self, input_data, axis=-2):
        import tensorflow as tf
        return tf.gather(input_data, self._indices, axis=axis)

    _identity = None

    @staticmethod
    def identity():
        if SkeletonConverter._identity is None:
            SkeletonConverter._identity = IdentityConverter()
        return SkeletonConverter._identity


class ExpandingSkeletonConverter(SkeletonConverter):
    def __init__(
            self, input_skeleton, output_skeleton, oi_map={}, empty_value=0):
        self._indices = tuple(get_transform_indices(
            output_skeleton, input_skeleton,
            {v: k for k, v in oi_map.items()}))
        self.input_skeleton = input_skeleton
        self.output_skeleton = output_skeleton
        self._empty_value = empty_value

    def convert(self, input_data, axis=-2):
        shape = list(input_data.shape)
        shape[axis] = self.output_skeleton.n_joints
        if self._empty_value is None:
            out = np.empty(shape, dtype=input_data.dtype)
        elif self._empty_value == 0:
            out = np.zeros(shape, dtype=input_data.dtype)
        else:
            out = np.ones(shape, dtype=input_data.dtype) * self._empty_value
        put(out, self._indices, input_data, axis=axis)
        return out

    def convert_tf(self, input_data, axis=-2):
        raise NotImplementedError()


class ExpandingMidJointSkeletonConverter(ExpandingSkeletonConverter):
    def __init__(
            self, input_skeleton, output_skeleton,
            mid_ids, left_ids, right_ids, oi_map={}, empty_value=None):
        super(ExpandingMidJointSkeletonConverter, self).__init__(
            input_skeleton, output_skeleton, oi_map, empty_value=empty_value)
        if isinstance(mid_ids, (str, unicode)):
            if not isinstance(left_ids, (str, unicode)):
                raise ValueError('left_ids must be a str if mid_ids is')
            if not isinstance(right_ids, (str, unicode)):
                raise ValueError('right_ids must be a str if mid_ids is')
            mid_ids = (mid_ids,)
            left_ids = (left_ids,)
            right_ids = (right_ids,)
        self._mids = tuple(output_skeleton.joint_index(m) for m in mid_ids)
        self._lefts = tuple(output_skeleton.joint_index(l) for l in left_ids)
        self._rights = tuple(output_skeleton.joint_index(r) for r in right_ids)

    def convert(self, input_data, axis=-2):
        out = super(ExpandingMidJointSkeletonConverter, self).convert(
            input_data, axis=axis)
        left_vals = np.take(out, self._lefts, axis=axis)
        right_vals = np.take(out, self._rights, axis=axis)
        if input_data.dtype == np.bool:
            mid_vals = np.logical_and(left_vals, right_vals)
        else:
            mid_vals = (left_vals + right_vals) / 2
        put(out, self._mids, mid_vals, axis=axis)
        return out

    def convert_tf(self, input_data, axis=-2, empty_value=None):
        raise NotImplementedError()


class IdentityConverter(SkeletonConverter):
    def __init__(self):
        pass

    def convert(self, input_data, axis=-2):
        return input_data

    def convert_tf(self, input_data, axis=-2):
        return input_data


class CompoundConverter(SkeletonConverter):
    def __init__(self, *converters):
        self._converters = converters
        self.input_skeleton = converters[0].input_skeleton
        self.output_skeleton = converters[-1].output_skeleton
        for i in range(len(converters) - 1):
            if converters[i].output_skeleton != converters[i+1].input_skeleton:
                raise ValueError('Invalid compound converter')

    def convert(self, input_data, axis=-2):
        for converter in self._converters:
            input_data = converter.convert(input_data, axis=axis)
        return input_data

    def convert_tf(self, input_data, axis=-2):
        for converter in self._converters:
            input_data = converter.convert_tf(input_data, axis=axis)
        return input_data
