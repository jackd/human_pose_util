from abc import abstractproperty
from abc import abstractmethod


def implements_dataset(obj):
    return all([hasattr(obj, n) for n in
                ['attrs', 'keys', 'items',
                 '__getitem__', '__contains__', '__iter__']])


class Dataset(object):
    """
    Minimal interface for sequential datasets used by parts of this package.

    Based on h5py groups/datasets. Objects should not be checked using
    isinstance since some will be other types, e.g. h5py groups. Use
    `implements_dataset(obj)` instead.
    """

    @abstractproperty
    def attrs(self):
        """Get an `AttributeManager` implementation."""
        raise NotImplementedError('Abstract property')

    @abstractmethod
    def keys(self):
        """Get a list of sequence keys."""
        raise NotImplementedError('Abstract method')

    def __contains__(self, key):
        """Test is key is in keys and hence usable in __getitem__."""
        return key in self.keys()

    @abstractmethod
    def __getitem__(self, key):
        """
        Get the example keyed by the given key.

        Should return a numpy array or Dataset implementation.
        """
        raise NotImplementedError('Abstract method')

    def __iter__(self):
        """Iterate over the keys."""
        return iter(self.keys())

    def items(self):
        """Iterate over all (key, value) pairs."""
        for k in self.keys():
            yield k, self[k]

    def __len__(self):
        """Get the number of keys in this dataset."""
        return len(self.keys())


class DatasetBase(Dataset):
    """Dataset based on maps."""
    def __init__(self, examples, attrs):
        if not isinstance(examples, dict):
            raise ValueError('Examples must be a dict.')
        self._examples = examples
        self.attrs = attrs

    def __getitem__(self, key):
        return self._examples[key]

    def keys(self):
        return self._examples.keys()

    def items(self):
        return self._examples.items()


class DelegatingDataset(Dataset):
    def __init__(self, base_dataset):
        self._base = base_dataset

    @property
    def base_dataset(self):
        return self._base

    @property
    def attrs(self):
        return self._base.attrs

    def keys(self):
        return self._base.keys()

    def _base_item(self, key):
        return self._base[key]

    def __contains__(self, key):
        return key in self._base

    def __getitem__(self, key):
        return self._base[key]


class MemoizingDataset(DelegatingDataset):
    def __init__(self, base_dataset):
        self._vals = {}
        super(MemoizingDataset, self).__init__(base_dataset)

    def __getitem__(self, key):
        if key not in self._vals:
            self._vals[key] = super(MemoizingDataset, self)[key]
        return self._vals[key]


def _to_hdf5(dataset, group):
    import numpy as np
    for k in dataset.attrs:
        group.attrs[k] = dataset.attrs[k]
    for k in dataset:
        v = dataset[k]
        if isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        else:
            _to_hdf5(v, group.create_group(k))


def to_hdf5(dataset, path):
    """Converts the dataset object to h5py."""
    import os
    import h5py
    if not implements_dataset(dataset):
        raise ValueError('dataset does not implement Dataset.')
    if os.path.isfile(path):
        raise ValueError('path %s already exists.')
    with h5py.File(path, 'w') as group:
        _to_hdf5(dataset, group)


def copy_dataset(dataset):
    """
    Deeply copies the given dataset into memory.

    Good for small datasets that have undergone extensive mapping and
    filtering.
    """
    import numpy as np
    if not implements_dataset(dataset):
        raise ValueError('dataset does not implement_dataset')
    examples = {k: v.copy() if isinstance(v, np.ndarray) else
                copy_dataset(v) for k, v in dataset.items()}

    attrs = {k: v for k, v in dataset.attrs.items()}
    return DatasetBase(examples, attrs)


class MappedDataset(DelegatingDataset):
    """Creates a mapped view of the base dataset."""
    def __init__(self, base_dataset, child_map_fn, attr_map_fn=None):
        """
        Create a mapped view of the base dataset.

        Args:
            base_dataset: original source of data
            child_map_fn: a function, or mapping keys of this to different
                values.
            attr_map_fn: dict of mapping function for attributes. See
                `MappedAttributeManager`.

        Returns:
            Dataset with modified values and attrs according to args.

            If child_map_fn is callable, then
                self[key] = child_map_fn(base_dataset[key])
            If child_map_fn

        """
        self._child_map_fn = child_map_fn
        if not callable(child_map_fn) and not all(
                [hasattr(child_map_fn, k)
                 for k in ['__contains__', '__getitem__']]):
            raise ValueError('child_map_fn must be callable or have '
                             '`__contains__`, `__getitem__` attributes')
        if attr_map_fn is None:
            self._attrs = base_dataset.attrs
        else:
            self._attrs = MappedAttributeManager(
                base_dataset.attrs, attr_map_fn)
        super(MappedDataset, self).__init__(base_dataset)

    def __getitem__(self, key):
        val = self._base_item(key)
        if callable(self._child_map_fn):
            return self._child_map_fn(val)
        elif key in self._child_map_fn:
            return self._child_map_fn[key](val)
        else:
            return val

    @property
    def attrs(self):
        return self._attrs


class FilteredDataset(DelegatingDataset):
    """Dataset with examples filtered."""
    def __init__(self, base_dataset, filter_fn):
        self._filter_fn = filter_fn
        super(FilteredDataset, self).__init__(base_dataset)

    def keys(self):
        return [k for k in super(FilteredDataset, self).keys()
                if self._filter_fn(self._base_item(k))]

    def __getitem__(self, key):
        val = self._base_item(key)
        if self._filter_fn(val):
            return val
        else:
            raise KeyError('Value for key %s failed filter predicate' % key)


def implements_attribute_manager(obj):
    return all([hasattr(obj, n) for n in
               ['keys', '__getitem__', '__contains__', '__iter__']])


class AttributeManager(object):
    """
    Minimal interface, based on h5py's AttributeManager class.

    Objects should not be checked using isinstance(obj, AttributeManager),
    since h5py has it's own implementation, and it's implemented by dicts
    anyway. Use `implements_attribute_manager(obj)` instead
    """
    @abstractmethod
    def keys():
        """Iterable of keys."""
        raise NotImplementedError('Abstract method')

    def __contains__(self, key):
        return key in self.keys()

    @abstractmethod
    def __getitem__(self, key):
        """Get the item keyed by the given key."""
        raise NotImplementedError('Abstract method')

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        for k in self.keys():
            yield k, self[k]


class MappedAttributeManager(AttributeManager):
    """AttributeManager that maps base values according to other values."""
    def __init__(self, base_manager, map_fn):
        """
        Create an AttributeManager that modifies a base.

        Args:
            base_manager: base AttributeManager
            map_fn: dict mapping selected attribute values.

        self[key] = map_fn[key](base_manager[key]) if key is in map_fn,
        otherwise returns the value unchanged. Contains all the same keys as
        the base manager.
        """
        self._base = base_manager
        self._map_fn = map_fn

    def __getitem__(self, key):
        val = self._base[key]
        return self._map_fn[key](val) if key in self._map_fn else val

    def keys(self):
        return self._base.keys()

    def __contains__(self, key):
        return key in self._base
