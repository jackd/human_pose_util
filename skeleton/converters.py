class SkeletonConverter(object):
    def __init__(self, input_skeleton, output_skeleton, oi_map={}):
        indices = []
        for i in range(output_skeleton.n_joints):
            joint = output_skeleton.joint(i)
            if joint in oi_map:
                joint = oi_map[joint]
            indices.append(input_skeleton.joint_index(joint))
        self._indices = indices
        self.input_skeleton = input_skeleton
        self.output_skeleton = output_skeleton

    @property
    def indices(self):
        return tuple(self._indices)

    def convert(self, input_data):
        return input_data[..., self._indices, :]

    def convert_tf(self, input_data):
        import tensorflow as tf
        return tf.gather(input_data, self._indices, axis=-2)

    _identity = None

    @staticmethod
    def identity():
        if SkeletonConverter._identity is None:
            SkeletonConverter._identity = IdentityConverter()
        return SkeletonConverter._identity


class IdentityConverter(SkeletonConverter):
    def __init__(self):
        pass

    def convert(self, input_data):
        return input_data

    def convert_tf(self, input_data):
        return input_data
