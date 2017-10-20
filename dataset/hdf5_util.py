import os
import numpy as np
import h5py


def _load_value(group, key):
    val = group[key]
    if isinstance(val, np.ndarray):
        return np.array(val)
    else:
        raise KeyError('Value at %s not numpy array' % key)


def _load_data(group, keys=[], attr_keys=[]):
    data = {}
    if isinstance(keys, (str, unicode)):
        data[keys] = _load_value(group, keys)
    for k in keys:
        data[k] = _load_value(group, k)

    if isinstance(attr_keys, (str, unicode)):
        data[k] = group.attrs[attr_keys]
    else:
        for k in attr_keys:
            data[k] = group.attrs[k]
    return data


def _save_data(group, values=None, attrs=None):
    if attrs is not None:
        for k, v in attrs.items():
            group.attrs[k] = v

    if values is not None:
        for k, v in values.items():
            if isinstance(v, np.ndarray):
                group.create_dataset(k, data=v)
            else:
                g = group.create_group(k)
                if hasattr(values, '__len__') and len(values) == 2:
                    _save_data(g, values[0], values[1])
                else:
                    _save_data(g, values)


def save_data(group_or_path, values, attrs):
    """
    Save data to the given group or path.

    Args:
        group_op_path: hdf5 group or path to hdf5 group.
        values: numpy array or dict mapping string keys to (values, attrs)
            structure.
    """
    if isinstance(group_or_path, (str, unicode)):
        path = group_or_path
        if os.path.isfile(path):
            raise IOError('File already exists at %s.' % path)
        with h5py.File(path, 'w') as f:
            _save_data(f, values, attrs)
    elif all([hasattr(group_or_path, k) for k in ['create_dataset', 'attrs']]):
        _save_data(group_or_path, values, attrs)
    else:
        raise TypeError('`group_or_path` must be h5py group-like or be a '
                        'string/unicode')


def load_data(group_or_path, keys, attr_keys):
    """
    Load data from the given group or path.

    Args:
        group_or_path: group-like (has `__getitem__` and `attrs` attributes)
            or path to a hdf5 file.
        keys: iterable of keys, or dict mapping string keys to tuples.
        attr_keys: iterable of keys

    Returns:
        dict mapping keys/attr_keys to copies of their values.

    Example usage:
    Assuming `path` is the path to a h5py file containing data at 'x', 'y' and
    attrs 'a0', 'a1':
    ```
    load_data(path, ['x'], ['a0'])  # {'x': ..., 'a0': ...}
    load_data(path, 'x')  # {'x': ...}
    load_data(path, 'x', 'a0'):  # {'x': ..., 'a0': ...}
    ```
    """
    if isinstance(group_or_path, (str, unicode)):
        with h5py.File(group_or_path) as f:
            return _load_data(f, keys, attr_keys)
    elif hasattr(group_or_path, '__getitem__', 'attrs'):
        return _load_data(group_or_path, keys, attr_keys)
    else:
        raise TypeError(
            'group_or_path must be str/unicode or have '
            '__getitem__/attrs attributes.')
