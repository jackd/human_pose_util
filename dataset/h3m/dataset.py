"""Wrapper for dataset satisfying specs."""
from __future__ import division
import os
import string
import numpy as np

from human_pose_util.dataset.interface import Dataset, AttributeManager
from human_pose_util.dataset.interface import MappedDataset
from human_pose_util.dataset.spec import scaled_dataset
from human_pose_util.dataset.spec import filter_by_camera
from human_pose_util.dataset.spec import modified_fps_dataset

from data import root_node

from human_pose_util.transforms.np_impl import euler_from_matrix_nh
from human_pose_util.transforms.camera_params import calculate_intrinsics
from human_pose_util.transforms.camera_params import calculate_extrinsics
from human_pose_util.transforms.np_impl import project
from human_pose_util.transforms.np_impl import transform_frame

print('********************************************************')
print('********************************************************')
print('********************DEPRECATED**************************')
print('**********human_pose_util.dataset.h3m.dataset***********')
print('Transition to human_pose_util.dataset.h3m.pose_sequence.')
print('********************************************************')
print('********************************************************')

_root_node = root_node()
_root_node.open()


def train_paths():
    return [str(v) for v in _root_node.views(include_eval=False)]


def eval_paths():
    return [str(v) for v in _root_node.views(include_train=False)]


class H3mDataset(Dataset):
    def __init__(self, training):
        self._training = training
        paths = train_paths() if training else eval_paths()
        self._keys = [p.replace('/', '__') for p in paths]
        self._keys.sort()
        self._keys = tuple(self._keys)
        self._attrs = {
            'skeleton_id': 's24',
        }

        self._examples = {
            k: H3mExample(k.replace('__', '/')) for k in self._keys}

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return self._keys

    def __contains__(self, key):
        return key in self._examples

    def __getitem__(self, key):
        return self._examples[key]


class H3mExample(Dataset):
    _keys = frozenset(['p3w', 'p3c', 'p2'])

    def __init__(self, view_path):
        _, subject_id, _, sequence_id, _, camera_id = \
            string.split(view_path, '/')
        self._view_path = view_path
        self._view = _root_node.view(view_path)
        self._sequence = self._view.sequence
        self._attrs = H3mExampleAttributeManager(
            self, subject_id, sequence_id, camera_id, self._view,
            self._sequence)

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return H3mExample._keys

    def __getitem__(self, key):
        if key == 'p3w':
            return np.array(self._sequence.p3, dtype=np.float32)
        elif key == 'p3c':
            return np.array(self._view.p3, dtype=np.float32)
        elif key == 'p2':
            return np.array(self._view.p2, dtype=np.float32)
        else:
            raise KeyError('key %s not recognized' % key)


class ConsistentH3mExample(H3mExample):
    def __init__(self, base_example):
        self._base = base_example
        self._p3c = None
        self._p2 = None
        self._attrs = base_example.attrs

    def __getitem__(self, key):
        if key == 'p3c':
            if self._p3c is None:
                self._calculate_p3c()
            return self._p3c
        elif key == 'p2':
            if self._p2 is None:
                self._calculate_p2()
            return self._p2
        else:
            return self._base[key]

    def _calculate_p3c(self):
        r, t = (self._base.attrs[k] for k in ('r', 't'))
        p3w = self._base['p3w']
        self._p3c = transform_frame(p3w, r, t)

    def _calculate_p2(self):
        f, c = (self._base.attrs[k] for k in ('f', 'c'))
        p3c = self['p3c']
        self._p2 = project(p3c, f=f, c=c)


def consistent_h3m_dataset(h3m_dataset):
    return MappedDataset(h3m_dataset, ConsistentH3mExample)


def s24_to_s14_dataset(h3m_dataset):
    from human_pose_util.skeleton import s24_to_s14_converter
    convert = s24_to_s14_converter().convert

    def mapped_example(example):
        return MappedDataset(
            example, {k: convert for k in ['p3c', 'p3w', 'p2']})

    return MappedDataset(
        h3m_dataset, mapped_example, {'skeleton_id': lambda x: 's14'})


class H3mExampleAttributeManager(AttributeManager):
    _keys = frozenset(
        ['subject_id', 'sequence_id', 'camera_id',
         'f', 'c', 't', 'r', 'n_frames', 'video_path', 'fps'])

    def __init__(
            self, example, subject_id, sequence_id, camera_id, view, sequence):
        self.attrs = {
            'subject_id': subject_id,
            'sequence_id': sequence_id,
            'camera_id': camera_id,
            'video_path': view.video_path,
            'fps': 50,
            'pixel_scale': 1,
            'space_scale': 1,
        }
        self._view = view
        self._sequence = sequence
        self._example = example

    def keys(self):
        return H3mExampleAttributeManager._keys

    def _calculate_intrinsics(self):
        self.attrs['f'], self.attrs['c'] = calculate_intrinsics(
            np.reshape(self._example['p3c'], (-1, 3)),
            np.reshape(self._example['p2'], (-1, 2)))

    def _calculate_extrinsics(self):
        p3w = self._example['p3w']
        p3c = self._example['p3c']
        R, t, k = calculate_extrinsics(
            np.reshape(p3w, (-1, 3)), np.reshape(p3c, (-1, 3)))
        assert(np.allclose(k, 1.0, atol=1e-3))
        self.attrs['r'] = np.array(euler_from_matrix_nh(R), dtype=np.float32)
        self.attrs['t'] = t

    def _calculate_n_frames(self):
        self.attrs['n_frames'] = len(self._example['p2'])

    def __getitem__(self, key):
        if key not in self.attrs:
            if key in ('f', 'c'):
                self._calculate_intrinsics()
            elif key in ('r', 't'):
                self._calculate_extrinsics()
            elif key == 'n_frames':
                self._calculate_n_frames()
            else:
                raise KeyError('Key %s not recognized' % key)
        return self.attrs[key]


def register_h3m_defaults():
    from skeleton import s24
    from human_pose_util.register import skeleton_register, dataset_register
    skeleton_register['s24'] = s24
    base = {
        'train': H3mDataset(True),
        'eval': H3mDataset(False)
    }
    consistent = {k: consistent_h3m_dataset(v) for k, v in base.items()}
    consistent_scaled = {k: scaled_dataset(v) for k, v in consistent.items()}
    camera_ids = list(
        set([v.attrs['camera_id'] for k, v in base['train'].items()]))
    camera_ids.sort()
    camera_id = camera_ids[1]
    consistent_scaled_c1 = {
        k: filter_by_camera(v, camera_id)
        for k, v in consistent_scaled.items()}
    dataset_register['h3m_base'] = base
    dataset_register['h3m_consistent'] = consistent
    dataset_register['h3m_consistent_scaled'] = consistent_scaled
    dataset_register['h3m_consistent_scaled_c1'] = consistent_scaled_c1
    s14_consistent_scaled_c1 = {
        k: filter_by_camera(
            scaled_dataset(
                s24_to_s14_dataset(v)), camera_id)
        for k, v in base.items()
    }
    dataset_register['h3m_s14_consistent_scaled_c1'] = s14_consistent_scaled_c1
    dataset_register['h3m_s14_consistent_scaled_c1_10fps'] = {
        k: modified_fps_dataset(v, 10)
        for k, v in s14_consistent_scaled_c1.items()
    }

    dataset_register['h3m_consistent_scaled_c1_10fps'] = {
        k: modified_fps_dataset(
            v, 10) for k, v in consistent_scaled_c1.items()}


def get_h3m_dataset(
        train=True, consistent=False, skeleton_id='s24', space_scale=1,
        pixel_scale=1, cameras=None, fps=None):
    """
    Get the h3m dataset with the specified parameters.

    Args:
        train: train or eval specifier
        consistent: if True, modifies p2 values such that projection(p3c) == p2
        skeleton_id: one of ['s24', 's14']
        space_scale: scaling factor applied to 3D positions
        pixel_scale: scaling factor applied to 2D positions
        cameras: list of indices of cameras to use.
        fps: changes frame rate if specified.
    """
    # examples = [_root_node[k] for k in _root_node.keys()]

    dataset = H3mDataset(train)
    if consistent:
        dataset = consistent(dataset)
    if skeleton_id == 's14':
        dataset = s24_to_s14_dataset(dataset)
    elif skeleton_id == 's24':
        pass
    else:
        raise ValueError('Invalid skeleton_id: %s' % skeleton_id)
    dataset = scaled_dataset(
        dataset, pixel_scale=pixel_scale, space_scale=space_scale)
    if cameras is not None:
        dataset = filter_by_camera(dataset, cameras)
    if fps is not None:
        dataset = modified_fps_dataset(dataset, fps)
    return dataset


_root_dir = os.path.realpath(os.path.dirname(__file__))


# def get_h3m_dataset_by_id(dataset_id, **kwargs):
#     """
#     Get a dataset with configuration specified in params directory.
#
#     See also: `get_h3m_dataset`
#     """
#     import json
#     # path = os.path.join(_root_dir, 'params', '%s.json' % dataset_id)
#     path = os.path.join(_root_dir, 'default_datasets.json')
#     if not os.path.isfile(path):
#         raise IOError(
#             'No params file for dataset %s at %s' % dataset_id, path)
#     with open(path, 'r') as f:
#         params = json.load(f)
#     params = params[dataset_id]
#     params.update(**kwargs)
#     return get_h3m_dataset(**params)


if __name__ == '__main__':
    # from human_pose_util.dataset.spec import pose_dataset_spec
    from human_pose_util.register import skeleton_register
    from human_pose_util.register import dataset_register
    from human_pose_util.dataset.eva.skeleton import s14
    from human_pose_util.transforms.np_impl import euler_matrix_nh
    from skeleton import s24
    skeleton_register['s24'] = s24
    skeleton_register['s14'] = s14
    register_h3m_defaults()
    # dataset = dataset_register['h3m_consistent_scaled_c1']['train']
    # pose_dataset_spec.assert_satisfied_by(dataset)
    dataset = dataset_register['h3m_consistent']['train']
    # dataset = dataset_register['h3m_consistent_scaled_c1']['train']
    # dataset = dataset_register['h3m_s14_consistent_scaled_c1']['train']
    # pose_dataset_spec.assert_satisfied_by(dataset)
    key = list(dataset.keys())[0]
    example = dataset[key]
    skeleton = skeleton_register[dataset.attrs['skeleton_id']]
    p2, p3w, p3c = [example[k][0] for k in ['p2', 'p3w', 'p3c']]
    r, t, f, c = [example.attrs[k] for k in ['r', 't', 'f', 'c']]
    Rt = euler_matrix_nh(*r).T
    # p3c_calc = transform_frame(p3w, r, t)
    p3c_calc = np.matmul(p3w, Rt) + t
    p2_calc = project(p3c_calc, f, c)
    print(np.max(np.abs(p2 - p2_calc)))
    print(np.max(np.abs(p2)))
