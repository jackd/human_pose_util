import string
import os
import numpy as np
from human_pose_util.dataset.normalize import \
    normalized_poses, normalize_sequences
from data import root_node
from human_pose_util.dataset.interface import Dataset, AttributeManager
from human_pose_util.transforms.np_impl import euler_from_matrix_nh
from human_pose_util.transforms.camera_params import calculate_intrinsics
from human_pose_util.transforms.camera_params import calculate_extrinsics

_root_node = root_node()
_root_node.open()
_root_dir = os.path.realpath(os.path.dirname(__file__))


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


def get_sequences(
        mode, camera_idxs=[1], dataset_attrs=['skeleton_id'],
        keys=['p3w'], attr_keys=[
            'f', 't', 'r', 'c', 'fps', 'subject_id', 'n_frames']):
    dataset = H3mDataset(mode)
    sequences = [dataset[k] for k in dataset]
    if camera_idxs is not None:
        camera_ids = list(
            set([v.attrs['camera_id'] for v in sequences]))
        camera_ids.sort()
        camera_ids = [camera_ids[i] for i in camera_idxs]
        sequences = [
            s for s in sequences if s.attrs['camera_id'] in camera_ids]

    dataset_attrs = {k: dataset.attrs[k] for k in dataset_attrs}
    data = []
    for sequence in sequences:
        s = dataset_attrs.copy()
        for key in keys:
            s[key] = sequence[key]
        for key in attr_keys:
            s[key] = sequence.attrs[key]
        data.append(s)
    return data


def get_normalized_p3(mode, skeleton_id='s24'):
    sequences = get_sequences(mode, attr_keys=['subject_id'])
    normalize_sequences(
            sequences, scale_to_height=True, target_skeleton_id=skeleton_id)
    p3 = np.concatenate([s['p3w'] for s in sequences], axis=0)
    p3 = normalized_poses(p3, skeleton_id, rotate_front=True, recenter_xy=True)
    return p3


def get_scaled_poses(
        mode, skeleton_id='s24', pixel_scale=1000, space_scale=1000,
        camera_idxs=[1], consistent_pose=True, consistent_projection=True,
        fps=50):
    keys = ['p3w']
    if not consistent_pose:
        keys.append('p3c')
    if not consistent_projection:
        keys.append('p2')
    sequences = get_sequences(mode, camera_idxs, keys=keys)
    normalize_sequences(
        sequences, consistent_pose=consistent_pose,
        consistent_projection=consistent_projection, space_scale=space_scale,
        pixel_scale=pixel_scale, fps=fps, target_skeleton_id=skeleton_id)
    p3w = np.concatenate([s['p3w'] for s in sequences], axis=0)
    p2 = np.concatenate([s['p2'] for s in sequences], axis=0)
    r, t, f, c = [np.concatenate(
        [[s[k] for i in range(s['n_frames'])] for s in sequences])
        for k in ['r', 't', 'f', 'c']]
    return p2, r, t, f, c, p3w


_pose_data_fns = {
    'normalized-p3': get_normalized_p3,
    'scaled-p2-p3': get_scaled_poses
}


def get_pose_data(problem_id, *args, **kwargs):
    return _pose_data_fns[problem_id](*args, **kwargs)


if __name__ == '__main__':
    from human_pose_util.serialization import register_datasets, \
        dataset_register
    register_datasets(h3m=True)
    for mode in ['train', 'eval']:
        for problem_id in ['normalized-p3', 'scaled-p2-p3']:
            # get_pose_data(problem_id, mode)
            dataset_register['h3m'](problem_id, mode)
    print('passed!')
