import string
import numpy as np
from human_pose_util.dataset.group import Group
from meta import eval_subjects, eval_base_sequences, cameras
from hdf5_tree import data_file, data_file_exists, RootNode, convert


if not data_file_exists():
    print('No hdf5 data exists: converting.')
    print('This may take some time...')
    convert()
assert(data_file_exists())
f = data_file('r')
_root_node = RootNode(f)


class _ModeKeys:
    train = 'train'
    eval = 'eval'


def train_ids():
    for subject in eval_subjects:
        for base_sequence in eval_base_sequences:
            for camera in cameras:
                for run in ['1', '3']:
                    yield string.join(
                        [subject, base_sequence, run, camera,
                         _ModeKeys.train], '_')


def eval_ids():
    for subject in eval_subjects:
        for base_sequence in eval_base_sequences:
            for camera in cameras:
                run = '1'
                yield string.join(
                    [subject, base_sequence, run, camera, _ModeKeys.eval],
                    '_')


def get_sequence(key):
    subject, base_sequence, run, camera, mode = key.split('_')
    return EvaSequence(subject, base_sequence, run, camera, mode)


class EvaDataset(Group):
    def __init__(self):
        possible_keys = tuple(train_ids()) + tuple(eval_ids())
        # Dirty hack - some keys have no data it seems...
        keys = []
        self[possible_keys[0]]
        for k in possible_keys:
            try:
                self[k]
                keys.append(k)
            except Exception:
                pass
        self._keys = tuple(keys)
        self._attrs = {
            'skeleton_id': 's20',
        }

    def keys(self):
        return self._keys

    @property
    def attrs(self):
        return self._attrs

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        return get_sequence(key)


def _valid(data, dtype):
    return np.array([k for k in data if np.all(np.isfinite(k))], dtype=dtype)


class EvaSequence(Group):
    _keys = frozenset(['p3w', 'p3c', 'p2'])

    def __init__(self, subject, base_sequence, run, camera, mode):
        sequence_id = '%s_%s' % (base_sequence, run)

        training = mode == _ModeKeys.train
        sequence = _root_node.sequence(subject, sequence_id)
        self._node = sequence.view(camera)
        partition = sequence.train_frame_partition
        if training:
            start_index = 0
            stop_index = partition[0]
        else:
            start_index, stop_index = partition
        f, c = self._node.camera_intrinsics
        r, t = self._node.camera_extrinsics
        # video_path = None if run == '3' else self._node.video_path
        try:
            video_path = self._node.video_path
        except IOError:
            video_path = None

        self._attrs = {
            'sequence_id': sequence_id,
            'base_sequence': base_sequence,
            'run': run,
            'camera_id': camera,
            'subject_id': subject,
            'start_index': start_index,
            'stop_index': stop_index,
            'video_path': video_path,
            'f': f.astype(np.float32),
            'r': r.astype(np.float32),
            'c': c.astype(np.float32),
            't': t.astype(np.float32),
            'fps': sequence.fps,
            'pixel_scale': 1,
            'space_scale': 1,
            'mode': mode,
            'n_frames': len(self._node.p2)
        }

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return EvaSequence._keys

    def __getitem__(self, key):
        if key == 'p3c':
            return _valid(self._node.p3_camera, dtype=np.float32)
        elif key == 'p3w':
            return _valid(self._node.p3_world, dtype=np.float32)
        elif key == 'p2':
            return _valid(self._node.p2, dtype=np.float32)
        else:
            raise KeyError('Unrecognized key: %s' % key)
