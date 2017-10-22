"""Provides functions/classes for ease of parameter serialization."""
from skeleton.converters import SkeletonConverter

print('************************')
print('************************')
print('************************')
print('DEPRECATED')
print('Use human_pose_util.register instead')
print('************************')
print('************************')
print('************************')


class ValidatingDict(dict):
    """Simple dict-like class that enforces a user-supplied validator_fn."""
    def __init__(self, validator_fn):
        """
        Create by specifying a validator_fn.

        validator_fn takes (self, key, value) and should raise an error if
        necessary.
        """
        self._validator_fn = validator_fn
        self._base = {}

    def __setitem__(self, key, value):
        self._validator_fn(self, key, value)
        self._base[key] = value
        return value

    def __getitem__(self, key):
        print(key, list(self.keys()))
        return self._base[key]

    def keys(self):
        return self._base.keys()

    def __contains__(self, key):
        return key in self._base


class FunctionDict(ValidatingDict):
    """
    Class for handling deserialization of functions.

    Keys must be strings/unicode and values must be callable.

    Overwriting a pre-registered function is not allowed.

    Example usage:
    ```

    def get_h3m_dataset(*args, **kwargs):
        ...

    dataset_register = FunctionDict()
    dataset_register['h3m'] = get_h3m_dataset

    args = [...],
    kwargs = {...}

    deserialized_dataset = dataset_register.call('h3m', *args, **kwargs)
    # same as
    deserialized_dataset = dataset_register['h3m'](*args, **kwargs)
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

    def call(self, key, *args, **kwargs):
        """Call the preregistered value function with args/kwargs."""
        return self[key](*args, **kwargs)


# dataset_register = FunctionDict()
# dataset_register_by_id = FunctionDict()


# def register_dataset_id_fn(dataset_id, params_path):
#     """
#     Registers a corresponding function in dataset_register_by_id.
#
#     Assumes params_path is a json file containing a dict mapping keys to
#     kwarg dicts, and that a base function has been registered with
#     `dataset_id` in `dataset_register`.
#     """
#     if not os.path.isfile(params_path):
#         raise IOError(
#             'No params file for dataset %s at %s' % dataset_id, params_path)
#
#     with open(params_path, 'r') as f:
#         params = json.load(f)
#
#     def get_dataset_by_id(key, **kwargs):
#         p = params[dataset_id].copy()
#         p.update(**kwargs)
#         return dataset_register[key](**p)
#
#     dataset_register_by_id[dataset_id] = get_dataset_by_id


def _dataset_validator(register, key, value):
    if key in register:
        raise KeyError('key already exists in dataset register: %s' % key)
    if not isinstance(key, (str, unicode)):
        raise KeyError('key must be string/unicode, got %s' % key)


dataset_register = ValidatingDict(_dataset_validator)


def _skeleton_registration_validator(register, key, value):
    if key in register and value != register[key]:
        raise KeyError('key already exists in skeleton register: %s' % key)
    if not isinstance(key, (str, unicode)):
        raise KeyError('key must be string/unicode, got %s' % key)
    # if not isinstance(value, Skeleton):
    #     raise ValueError('value must be Skeleton, got %s' % value)
    converter_register[(key, key)] = SkeletonConverter.identity()


skeleton_register = ValidatingDict(_skeleton_registration_validator)


def _converter_registration_validator(register, key, value):
    if key in register and value != register[key]:
        raise KeyError('key already exists in skeleton register: %s' % key)
    if not isinstance(key, tuple) and all(
            [isinstance(k, (str, unicode)) for k in key]):
        raise KeyError('key must be a tuple of str/unicodes, got %s' % key)
    # if not isinstance(value, SkeletonConverter):
    #     raise ValueError('value must be a SkeletonConverter, got %s' % value)
    if key[0] != key[1]:
        if value.input_skeleton != skeleton_register[key[0]]:
            raise ValueError('Skeleton id mismatch: input_skeleton')
        if value.output_skeleton != skeleton_register[key[1]]:
            raise ValueError('Skeleton id mismatch: output_skeleton')


converter_register = ValidatingDict(_converter_registration_validator)


def register_skeletons(h3m=False, eva=False, mpi_inf=False):
    if h3m:
        from dataset.h3m.skeleton import s24
        skeleton_register['s24'] = s24
    if eva:
        from dataset.eva.skeleton import s14, s16, s20
        skeleton_register['s14'] = s14
        skeleton_register['s16'] = s16
        skeleton_register['s20'] = s20
    if mpi_inf:
        from dataset.mpi_inf.skeleton import base, relevant, extended
        skeleton_register['mpi-inf_base'] = base
        skeleton_register['mpi-inf_extended'] = extended
        skeleton_register['mpi-inf_relevant'] = relevant


def register_datasets(h3m=False, eva=False, mpi_inf=False):
    register_skeletons(h3m=h3m, eva=eva, mpi_inf=mpi_inf)
    if h3m:
        from dataset.h3m.dataset import dataset
        dataset_register['h3m'] = dataset
    if eva:
        raise NotImplementedError()
    if mpi_inf:
        raise NotImplementedError()


def register_converters(h3m_eva=False):
    import skeleton.converters as c
    if h3m_eva:
        _oi_map = {'head': 'head-back'}
        s24, s14, s16 = [skeleton_register[k] for k in ['s24', 's14', 's16']]
        converter_register[('s24', 's14')] = c.SkeletonConverter(
            s24, s14, _oi_map)
        # converter_register[('s24', 's16')] = c.SkeletonConverter(
        #     s24, s16, _oi_map)  #  s24 has no no pelvis :(


def get_skeleton_register():
    return skeleton_register


if __name__ == '__main__':
    from dataset.normalize import normalized_view_data, normalized_p3w
    register_skeletons(h3m=True, eva=True, mpi_inf=True)
    register_datasets(h3m=True)
    register_converters(h3m_eva=True)
    print('Registration successful!')

    dataset = dataset_register['h3m']
    for mode in ['train', 'eval']:
        print('Getting normalized_view_data...')
        normalized_view_data(dataset, modes=mode)

        print('Getting normalized_p3w...')
        print(skeleton_register.keys())
        normalized_p3w(dataset, modes=mode)
