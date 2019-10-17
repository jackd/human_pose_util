"""
Provides Group/AttributeManager interfaces.

Base on (and implemented by) hdf5 `Group`s/`AttributeManager`s.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from abc import abstractproperty, abstractmethod


def implements_group(obj, check_children=False, deep=False):
    """Test whether the given object satisfies the Dataset interface."""
    if not all([
            hasattr(obj, n) for n in [
                'attrs', 'keys', 'items', '__getitem__', '__setitem__',
                '__contains__', '__iter__'
            ]
    ]):
        return False
    if check_children:
        for v in obj.values():
            if not (isinstance(v, np.ndarray) or
                    implements_group(v, check_children=deep, deep=deep)):
                return False
    return True


class Group(object):
    """
    Minimal interface for sequential datasets used by parts of this package.

    Based on h5py groups/datasets. Objects should not be checked using
    isinstance since some will be other types, e.g. h5py groups. Use
    `implements_group(obj)` instead.
    """

    @abstractproperty
    def attrs(self):
        """Get an `AttributeManager` implementation."""
        raise NotImplementedError('Abstract property')

    @abstractmethod
    def keys(self):
        """Get a list of sequence keys."""
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def __getitem__(self, key):
        """
        Get the example keyed by the given key.

        Should return a numpy array or Dataset implementation.
        """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def __setitem__(self, key, value):
        """
        Set the data keyed by the given key.

        Some groups may be immutable. Consider using `copy_group` in this case.
        """
        raise NotImplementedError('Abstract method')

    def __contains__(self, key):
        """Test is key is in keys and hence usable in __getitem__."""
        return key in self.keys()

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

    def values(self):
        return (self[k] for k in self.keys())


class CombinedGroup(Group):
    """Class combining a dict of groups/data and AttributeManager/dict."""

    def __init__(self, group_dict, attrs=None):
        self._group_dict = group_dict
        self._attrs = {} if attrs is None else attrs

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return self._group_dict.keys()

    def __contains__(self, key):
        return key in self._group_dict

    def __getitem__(self, key):
        return self._group_dict[key]

    def __setitem__(self, key, value):
        self._group_dict[key] = value

    def items(self):
        return self._group_dict.items()

    def __len__(self):
        return len(self._group_dict)


class KeySubsetGroup(Group):
    """
    Group exposing only a subset of values of an underlying group.

    e.g. if group = {'a': 'hello', 'b': 'world'}
    then KeySubsetGroup(group, ['a']) is {'a': 'hello'}, attrs={}
    """

    def __init__(self, group, keys=None, attr_keys=None):
        """
        Create with a backing group/dict, keys and attr_keys to include.

        Args:
            group: Group or dict containing all of the keys specified in keys.
            keys: iterable of keys to include. Should support fast `include`,
                e.g. short list/tuple, set. If None, all keys in group are
                included.
            attr_keys: attribute keys to include. If none, includes all
                attributes of group if None, or creates an empty dict if
                group doesn't have an attrs attribute.
        """
        if keys is None:
            self._keys = group.keys()
        else:
            for k in keys:
                if k not in group:
                    raise KeyError('KeySubsetGroup keys must all be in group, '
                                   'but %s is not.' % k)
            self._keys = keys
        if not hasattr(group, 'attrs'):
            if attr_keys is None:
                self._attrs = {}
            else:
                raise ValueError('attr_keys not allowed if group has no attrs')
        else:
            self._attrs = group.attrs if attr_keys is None else \
                {k: group.attrs[k] for k in attr_keys}
        self._group = group

    def keys(self):
        return self._keys

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError('key not in subset: %s' % key)
        return self._group[key]

    def __contains__(self, key):
        return key in self._keys

    def items(self):
        for key in self.keys():
            yield key, self._group[key]

    def values(self):
        for key in self.keys():
            yield self._group[key]


# class DelegatingGroup(Group):
#     """
#     Base class that delegates all actions to a backing group.
#
#     While the implementation is not abstract, instantiating an element of
#     this class will be indistinguishable from the original group.
#     """
#     def __init__(self, group):
#         """Initialize with a data dict-like data and attrs objects."""
#         self._group = group
#
#     @property
#     def attrs(self):
#         """Get the corresponding attributes: generally smaller data."""
#         return self._group.attrs
#
#     def keys(self):
#         """Get the keys of this Group."""
#         return self._group.keys()
#
#     def __contains__(self, key):
#         return key in self._group
#
#     def __getitem__(self, key):
#         """Get the value of the corresponding key."""
#         return self._group[key]
#
#     def __setitem__(self, key, value):
#         self._group[key] = value
#
#     def __len__(self):
#         return len(self._group)

# class FilteredGroup(DelegatingGroup):
#     def __init__(self, group, filter_fn):
#         """
#         Filter a group by the specified filter_fn.
#
#         `filter_fn` must be a function mapping (key, value) -> bool
#         """
#         self._filter_fn = filter_fn
#         super(FilteredGroup, self).__init__(group)
#
#     def keys(self):
#         return (k for k in super(FilteredGroup, self).keys()
#                 if self._filter_fn(k, self[k]))
#
#     def __contains__(self, key):
#         return super(FilteredGroup, self).__contains__(key) and \
#             self._filter_fn(key, super(FilteredGroup, self).__getitem__(key))
#
#     def __getitem__(self, key):
#         val = super(FilteredGroup, self)[key]
#         if not self._filter_fn(key, val):
#             raise KeyError('Predicate test failed: %s' % str((key, val)))
#         return val
#
#     def items(self):
#         for k, v in super(FilteredGroup, self).items():
#             if self._filter_fn(k, v):
#                 yield k, v
#
#     def __len__(self):
#         return len(self.items())


class MappedDict(object):
    """
    dict-like class where values are lazily evaluated.

    Immutable, so long as the base dict is immutable.
    """

    def __init__(self, base, map_fn):
        self._base = base
        self._map_fn = map_fn

    def keys(self):
        return self._base.keys()

    def __contains__(self, key):
        return key in self._base

    def __getitem__(self, key):
        return self._map_fn(self._base[key])

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __iter__(self):
        """Key iterator."""
        return iter(self.keys())

    def __len__(self):
        return len(self._base)


def mapped_group(group, map_fn, attr_map_fn=None):
    if hasattr(group, 'attrs'):
        attrs = group.attrs if attr_map_fn is None else \
            MappedDict(group.attrs, attr_map_fn)
    else:
        if attr_map_fn is None:
            attrs = {}
        else:
            raise ValueError('group has no attrs, so attr_map_fn must be None')
    group = MappedDict(group, map_fn)
    return CombinedGroup(group, attrs)


def filter_children(group, item_filter_fn):
    keys = set([k for k, v in group.items() if item_filter_fn(k, v)])
    return KeySubsetGroup(group, keys)


class AttributeManager(object):
    """
    Minimal interface, based on h5py's AttributeManager class.

    Objects should not be checked using isinstance(obj, AttributeManager),
    since h5py has it's own implementation, and it's implemented by dicts
    anyway. Use `implements_attribute_manager(obj)` instead
    """

    @abstractmethod
    def keys(self):
        """Get an iterable of keys."""
        raise NotImplementedError('Abstract method')

    def __contains__(self, key):
        """Flag whether this[key] has a value."""
        return key in self.keys()

    @abstractmethod
    def __getitem__(self, key):
        """Get the item keyed by the given key."""
        raise NotImplementedError('Abstract method')

    def __iter__(self):
        """Key iterator."""
        return iter(self.keys())

    def items(self):
        """Get all key, value pairs associated with this manager."""
        for k in self.keys():
            """Iterator for all key, value pairs."""
            yield k, self[k]

    def values(self):
        return (self[k] for k in self.keys())


def implements_attribute_manager(obj):
    """Test whether the target implements the AttributeManager interface."""
    return all([
        hasattr(obj, n)
        for n in ['keys', '__getitem__', '__contains__', '__iter__', 'items']
    ])


def copy_attrs(attrs):
    """
    Create a new dict from attrs.

    The returned dict is a copy, but values will not be copied.
    """
    return {k: v for k, v in attrs.items()}


def copy_group(group):
    return CombinedGroup({k: copy(v) for k, v in group.items()},
                         copy_attrs(group.attrs))


def copy(group_or_data):
    if isinstance(group_or_data, np.ndarray):
        return group_or_data.copy()
    else:
        return CombinedGroup({k: v for k, v in group_or_data.items()},
                             copy_attrs(group_or_data.attrs))
