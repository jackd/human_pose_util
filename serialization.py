"""Provides functions/classes for ease of parameter serialization."""
import os
import json
from skeleton import Skeleton


class ValidatingDict(dict):
    """Simple dict wrapper class that enforces a user-supplied validator_fn."""
    def __init__(self, validator_fn):
        """
        Create by specifying a validator_fn.

        validator_fn takes (self, key, value) and should raise an error if
        necessary.
        """
        self._validator_fn = validator_fn

    def __setitem__(self, key, value):
        self._validator_fn(self, key, value)
        super(ValidatingDict, self)[key] = value
        return value


class FunctionDict(ValidatingDict):
    """
    Class for handling deserialization of functions.

    Example usage:
    ```

    def get_h3m_dataset(*args, **kwargs):
        ...

    dataset_register = FunctionDict()
    dataset_register['h3m'] = get_h3m_dataset

    dataset_params = {
        'key': 'h3m',
        'args': [...],
        'kwargs': {...}
    }

    deserialized_dataset = dataset_register.call(**dataset_params)
    ```
    """
    def __init__(
            self, allow_overwrite=False, string_ids=True, val_callable=True):
        def validator_fn(self, key, val):
            if not allow_overwrite and key in self:
                raise KeyError('Key already present: cannot override values.')
            if string_ids and not isinstance(key, (str, unicode)):
                raise KeyError('Key must be string or unicode')
            if val_callable and not callable(val):
                raise ValueError('val must be callable.')
        super(FunctionDict, self).__init__(validator_fn)

    def call(self, key, args=[], kwargs={}):
        """Call the preregistered value function with args/kwargs."""
        return self[key](*args, **kwargs)


dataset_register = FunctionDict()
dataset_register_by_id = FunctionDict()


def register_dataset_id_fn(dataset_id, params_path):
    """
    Registers a corresponding function in dataset_register_by_id.

    Assumes params_path is a json file containing a dict mapping keys to
    kwarg dicts, and that a base function has been registered with
    `dataset_id` in `dataset_register`.
    """
    if not os.path.isfile(params_path):
        raise IOError(
            'No params file for dataset %s at %s' % dataset_id, params_path)

    def get_dataset_by_id(key, **kwargs):
        with open(params_path, 'r') as f:
            params = json.load(f)
        params = params[dataset_id]
        params.update(**kwargs)
        return dataset_register[key](**params)

    dataset_register_by_id[dataset_id] = get_dataset_by_id


def _skeleton_registration_validator(register, key, value):
    if key in register:
        raise KeyError('key already exists in skeleton register: %s' % key)
    if not isinstance(key, (str, unicode)):
        raise KeyError('key must be string/unicode, got %s' % key)
    if not isinstance(value, Skeleton):
        raise ValueError('value must be Skeleton, got %s' % value)


skeleton_register = ValidatingDict(_skeleton_registration_validator)
