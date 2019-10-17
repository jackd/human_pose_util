"""Provides interface for models that can be evaluated."""
from abc import abstractmethod, abstractproperty


class Model(object):
    _constructors = {}

    @staticmethod
    def register_constructor(model_type, constructor):
        if model_type in Model._constructors:
            raise Exception(
                'Constructor already registered for %s' % model_type)
        Model._constructors[model_type] = constructor

    @staticmethod
    def from_constructor(model_type, model_id):
        if model_type not in Model._constructors:
            raise Exception('Constructor not registered for %s' % model_type)
        return Model._constructors[model_type](model_id)

    @abstractproperty
    def model_type(self):
        """An identifier for the type of model."""
        pass

    @abstractproperty
    def model_id(self):
        """
        Get an identifier for the model.

        Must be unique for a given `model_type`.
        """
        pass

    @abstractproperty
    def skeleton(self):
        """Get the skeleton ID for the model."""
        pass

    @abstractproperty
    def observation_keys(self):
        """
        Get the observation keys used by this model.

        Must be an iterable containing any of the following:
            'p2': 2d poses, (n_frames, n_joints, 2), in pixel coordinates
            'f': camera projection (2,), in pixel coordinates
            'c': camera projection offset (2,), in pixel coordinates
            't': camera offset (3,), relative to world coordinates
            'r': camera rotation (3,), relative to world coordinates
        """
        pass

    @abstractproperty
    def inference_keys(self):
        """
        Get the keys for inferences producted by this model.

        Must be an iterable containing any of the following:
            'p3w': 3d pose in world coordinates (n_frames, n_joints, 3) in mm
            'p3c': 3d pose in camera coordinates, (n_frames, n_joints, 3) in mm
        """
        pass

    @abstractmethod
    def infer(self, observations):
        """
        Infer the pose given observation data.

        observations is a dict mapping keys from observation_keys to a numpy
        array.

        Returns a dict mapping keys in `inference_keys` to values.
        """
        pass
