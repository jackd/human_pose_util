import string
import numpy as np
from human_pose_util.dataset.interface import Dataset
from human_pose_util.dataset.spec import scaled_dataset, filter_by_camera
from meta import eval_subjects, eval_base_sequences, cameras
from hdf5_tree import data_file, data_file_exists, RootNode, convert
from human_pose_util.dataset.interface import MappedDataset
from human_pose_util.dataset.spec import modified_fps_dataset
from skeleton import s20_to_s14_converter, s20_to_s16_converter


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


def get_example(example_id):
    subject, base_sequence, run, camera, mode = example_id.split('_')
    return HumanEva1Example(subject, base_sequence, run, camera, mode)


class HumanEva1Dataset(Dataset):
    def __init__(self, training):
        self._training = training
        # Dirty hack - some keys have no data it seems...
        possible_keys = tuple(train_ids() if training else eval_ids())
        keys = []
        for k in possible_keys:
            try:
                self[k]
                keys.append(k)
            except Exception:
                pass
        self._keys = tuple(keys)
        self._attrs = {
            'skeleton_id': 's20',
            'pixel_scale': 1,
            'space_scale': 1,
        }

    def keys(self):
        return self._keys

    @property
    def attrs(self):
        return self._attrs

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        return get_example(key)


def _valid(data, dtype):
    return np.array([k for k in data if np.all(np.isfinite(k))], dtype=dtype)


class HumanEva1Example(Dataset):
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
            'mode': mode,
            'fps': sequence.fps
        }

    @property
    def attrs(self):
        return self._attrs

    def keys(self):
        return HumanEva1Example._keys

    def __getitem__(self, key):
        if key == 'p3c':
            return _valid(self._node.p3_camera, dtype=np.float32)
        elif key == 'p3w':
            return _valid(self._node.p3_world, dtype=np.float32)
        elif key == 'p2':
            return _valid(self._node.p2, dtype=np.float32)
        else:
            raise KeyError('Unrecognized key: %s' % key)


def _s20_to_other_dataset(eva_dataset, converter, target_skeleton_id):
    convert = converter.convert

    def mapped_example(example):
        return MappedDataset(
            example, {k: lambda ex: convert(ex[k])
                      for k in ['p3c', 'p3w', 'p2']})

    return MappedDataset(
        eva_dataset, mapped_example,
        {'skeleton_id': lambda ex: target_skeleton_id})


def s20_to_s16_dataset(eva_dataset):
    return _s20_to_other_dataset(eva_dataset, s20_to_s16_converter(), 's16')


def s20_to_s14_dataset(eva_dataset):
    return _s20_to_other_dataset(eva_dataset, s20_to_s14_converter(), 's14')


def register_eva_defaults():
    from skeleton import s14, s16, s20
    from human_pose_util.register import skeleton_register, dataset_register
    for key, value in [['s14', s14], ['s16', s16], ['s20', s20]]:
        skeleton_register[key] = value
    base = {
        'train': HumanEva1Dataset(True),
        'eval': HumanEva1Dataset(False),
    }
    scaled = {k: scaled_dataset(v) for k, v in base.items()}
    camera_ids = list(
        set([v.attrs['camera_id'] for k, v in base['train'].items()]))
    camera_ids.sort()
    camera_id = camera_ids[1]
    scaled_c1 = {k: filter_by_camera(v, camera_id) for k, v in scaled.items()}
    s16_scaled_c1 = {
        k: filter_by_camera(
            scaled_dataset(
                s20_to_s14_dataset(v)), camera_id)
        for k, v in base.items()
    }
    s14_scaled_c1 = {
        k: filter_by_camera(
            scaled_dataset(
                s20_to_s14_dataset(v)), camera_id)
        for k, v in base.items()
    }

    dataset_register['eva_base'] = base
    dataset_register['eva_scaled'] = scaled
    dataset_register['eva_scaled_c1'] = scaled_c1
    dataset_register['eva_scaled_c1_10fps'] = {
        k: modified_fps_dataset(v, 10) for k, v in scaled_c1.items()
    }
    # s16
    dataset_register['eva_s16_scaled_c1'] = s16_scaled_c1
    dataset_register['eva_s16_scaled_c1_10fps'] = {
        k: modified_fps_dataset(v, 10) for k, v in s16_scaled_c1.items()
    }
    # s14
    dataset_register['eva_s14_scaled_c1'] = s14_scaled_c1
    dataset_register['eva_s14_scaled_c1_10fps'] = {
        k: modified_fps_dataset(v, 10) for k, v in s14_scaled_c1.items()
    }
