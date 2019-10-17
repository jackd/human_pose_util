"""
See https://stackoverflow.com/questions/1977362/
    how-to-create-module-wide-variables-in-python:
Explicit access to module level variables by accessing them explicity on the
module
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# this = sys.modules[__name__]
#
from human_pose_util.skeleton.converters import SkeletonConverter
_skeleton_register = {}


def register_skeleton(skeleton_id, skeleton):
    _skeleton_register[skeleton_id] = skeleton


def get_skeleton(skeleton_id):
    return _skeleton_register[skeleton_id]


_dataset_register = {}


def register_dataset(dataset_id, dataset):
    _dataset_register[dataset_id] = dataset


def get_dataset(dataset_id):
    return _dataset_register[dataset_id]


_converter_register = {}


def register_converter(k0, k1, converter):
    _converter_register[(k0, k1)] = converter


def get_converter(k0, k1):
    if k0 == k1:
        return SkeletonConverter.identity()
    return _converter_register[(k0, k1)]


def register_skeletons(h3m=False, eva=False, mpi_inf=False):
    if h3m:
        from human_pose_util.dataset.h3m.skeleton import s24
        register_skeleton('s24', s24)
    if eva:
        from human_pose_util.dataset.eva.skeleton import s14, s16, s20
        from human_pose_util.dataset.eva.skeleton import s20_to_s14_converter
        from human_pose_util.dataset.eva.skeleton import s20_to_s16_converter
        for k, s in [('s14', s14), ('s16', s16), ('s20', s20)]:
            register_skeleton(k, s)
        for k0, k1, converter in [['s20', 's14',
                                   s20_to_s14_converter()],
                                  ['s20', 's16',
                                   s20_to_s16_converter()]]:
            register_converter(k0, k1, converter)
    if mpi_inf:
        from human_pose_util.dataset.mpi_inf.skeleton import relevant, base, extended
        for k, s in [('mpi-inf-base', base), ('mpi-inf-relevant', relevant),
                     ('mpi-inf-extended', extended)]:
            register_skeleton(k, s)


def register_datasets(h3m=False, eva=False, mpi_inf=False):
    register_skeletons(h3m=h3m, eva=eva, mpi_inf=mpi_inf)
    if h3m:
        from human_pose_util.dataset.h3m.dataset import H3mDataset
        register_dataset('h3m', H3mDataset())
    if eva:
        from human_pose_util.dataset.eva.dataset import EvaDataset
        register_dataset('eva', EvaDataset())
    if mpi_inf:
        raise NotImplementedError()


# class Register(object):
#     """Thin wrapper around a dictionary that ensures no overwriting."""
#     def __init__(self, validator_fn=None):
#         self._registry = {}
#         self._validator_fn = validator_fn
#
#     def __setitem__(self, key, value):
#         if key in self._registry:
#             if self._registry[key] != value:
#                 raise KeyError(
#                     'Different value already registered for %s' % key)
#         else:
#             if self._validator_fn is None or self._validator_fn(value):
#                 self._registry[key] = value
#             else:
#                 raise ValueError('Failed validation: %s' % key)
#
#     def __getitem__(self, key):
#         if key not in self._registry:
#             raise KeyError('No value registered for key %s' % key)
#         return self._registry[key]
#
#     def __delete__(self, key):
#         del self._registry[key]
#
#     def __contains__(self, key):
#         return key in self._registry
#
#     def keys(self):
#         return self._registry.keys()
#
#     def items(self):
#         return self._registry.items()
#
#
# dataset_register = Register()
# skeleton_register = Register()
# registers = Register()
# registers['skeleton'] = skeleton_register
# registers['dataset'] = dataset_register
#
#
# def register_default_datasets():
#     from dataset.h3m.dataset import register_h3m_defaults
#     from dataset.eva.dataset import register_eva_defaults
#     register_h3m_defaults()
#     register_eva_defaults()
#
#
# def register_default_skeletons():
#     from dataset.h3m.skeleton import s24
#     from dataset.eva.skeleton import s14, s16, s20
#     # h3m skeleton
#     skeleton_register['s24'] = s24
#     # eva skeletons
#     for k, v in [['s20', s20], ['s16', s16], ['s14', s14]]:
#         skeleton_register[k] = v
#
#
# def register_defaults():
#     register_default_skeletons()
#     register_default_datasets()
#
#
# if __name__ == '__main__':
#     register_defaults()
