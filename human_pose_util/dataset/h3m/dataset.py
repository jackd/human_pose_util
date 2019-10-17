from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from human_pose_util.dataset.h3m.data import root_node
from human_pose_util.dataset.group import Group, AttributeManager
from human_pose_util.transforms.np_impl import euler_from_matrix_nh
from human_pose_util.transforms.camera_params import calculate_intrinsics
from human_pose_util.transforms.camera_params import calculate_extrinsics

_root_node = root_node()
_root_node.open()
_root_dir = os.path.realpath(os.path.dirname(__file__))

# def _get_train_paths():
#     return [str(v) for v in _root_node.views(include_eval=False)]

# def _eval_paths():
#     return [str(v) for v in _root_node.views(include_train=False)]

_train_paths = set((str(v) for v in _root_node.views(include_eval=False)))
_eval_paths = set((str(v) for v in _root_node.views(include_train=False)))


class H3mDataset(Group):

    def __init__(self):
        paths = list(_train_paths) + list(_eval_paths)
        paths.sort()
        self._keys = [p.replace('/', '__') for p in paths]
        self._attrs = {'skeleton_id': 's24'}

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, key):
        return H3mSequence(key.replace('__', '/'))

    def keys(self):
        return self._keys


class H3mSequence(Group):
    """
    Group implementation for sequence data.

    attrs are common to all poses.
    """

    _keys = frozenset(['p3w', 'p3c', 'p2'])

    def __init__(self, view_path):
        """
        Initialize with the view path.

        subjects/subject_id/sequences/sequence_id/views/camera_id
        """
        _, subject_id, _, sequence_id, _, camera_id = view_path.split('/')
        self._view_path = view_path
        self._view = _root_node.view(view_path)
        self._sequence = self._view.sequence
        mode = 'train' if view_path in _train_paths else 'eval'
        self._attrs = SequenceAttributeManager(self, subject_id, sequence_id,
                                               camera_id, self._view,
                                               self._sequence, mode)

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return H3mSequence._keys

    def __getitem__(self, key):
        if key == 'p3w':
            return np.array(self._sequence.p3, dtype=np.float32)
        elif key == 'p3c':
            return np.array(self._view.p3, dtype=np.float32)
        elif key == 'p2':
            return np.array(self._view.p2, dtype=np.float32)
        else:
            raise KeyError('key %s not recognized' % key)


class SequenceAttributeManager(AttributeManager):
    _keys = frozenset([
        'subject_id', 'sequence_id', 'camera_id', 'f', 'c', 't', 'r',
        'n_frames', 'video_path', 'fps', 'mode'
    ])

    def __init__(self, example, subject_id, sequence_id, camera_id, view,
                 sequence, mode):
        self.attrs = {
            'subject_id': subject_id,
            'sequence_id': sequence_id,
            'camera_id': camera_id,
            'video_path': view.video_path,
            'fps': 50,
            'pixel_scale': 1,
            'space_scale': 1,
            'mode': mode
        }
        self._view = view
        self._sequence = sequence
        self._example = example

    def keys(self):
        return SequenceAttributeManager._keys

    def _calculate_intrinsics(self):
        self.attrs['f'], self.attrs['c'] = calculate_intrinsics(
            np.reshape(self._example['p3c'], (-1, 3)),
            np.reshape(self._example['p2'], (-1, 2)))

    def _calculate_extrinsics(self):
        p3w = self._example['p3w']
        p3c = self._example['p3c']
        R, t, k = calculate_extrinsics(np.reshape(p3w, (-1, 3)),
                                       np.reshape(p3c, (-1, 3)))
        assert (np.allclose(k, 1.0, atol=1e-3))
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


if __name__ == '__main__':
    from human_pose_util.register import register_datasets, get_dataset
    register_datasets(h3m=True)
    for mode in ['train', 'eval']:
        for problem_id in ['normalized-p3', 'scaled-p2-p3']:
            # get_pose_data(problem_id, mode)
            print(
                get_dataset('h3m')
                ['subjects__S9__sequences__Walking__views__60457274'])
    print('passed!')
