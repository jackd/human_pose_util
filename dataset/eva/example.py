"""TreeNode -> Dataset adapter."""
import string
from human_pose_util.dataset import PoseSequenceExample, pose_sequence_keys
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


def get_example(example_id):
    subject, base_sequence, run, camera, mode = example_id.split('_')
    return _HumanEva1Example(subject, base_sequence, run, camera, mode)


# class _HumanEva1Dataset(PoseSequenceDataset):
#     def __init__(self, root):
#         self._root = root
#
#     @property
#     def train_ids(self):
#
#
#     @property
#     def eval_ids(self):
#
#
#     def get_example(self, id_):
#         subject, base_sequence, run, camera, mode = id_.split('_')
#         return _HumanEva1Example(
#             self._root, subject, base_sequence, run, camera, mode)

human_eva1_keys = pose_sequence_keys.union(frozenset(['base_sequence', 'run']))


class _HumanEva1Example(PoseSequenceExample):
    def __init__(self, subject, base_sequence, run, camera, mode):
        sequence = '%s_%s' % (base_sequence, run)
        training = mode == _ModeKeys.train
        self._mode = mode
        sequence = _root_node.sequence(subject, sequence)
        self._node = sequence.view(camera)
        partition = sequence.train_frame_partition
        if training:
            self._start_index = 0
            self._stop_index = partition[0]
        else:
            self._start_index, self._stop_index = partition
        self._f, self._c = self._node.camera_intrinsics
        self._r, self._t = self._node.camera_extrinsics

        self._subject = subject
        self._base_sequence = base_sequence
        self._run = run
        self._camera_id = camera

    def mode(self):
        return self._mode

    def example_id(self):
        return string.join(
            [self.subject_id, self.base_sequence, self.run, self.camera_id,
             self.mode], '_')

    def keys(self):
        return human_eva1_keys

    @property
    def subject_id(self):
        return self._subject

    @property
    def base_sequence(self):
        return self._base_sequence

    @property
    def action_id(self):
        return self._base_sequence

    @property
    def run(self):
        return self._run

    @property
    def camera_id(self):
        return self._camera

    @property
    def p2(self):
        return self._node.p2[self._start_index: self._stop_index]

    @property
    def p3w(self):
        return self._node.p3_world[self._start_index: self._stop_index]

    @property
    def p3c(self):
        return self._node.p3_camera[self._start_index: self._stop_index]

    @property
    def r(self):
        return self._r

    @property
    def t(self):
        return self._t

    @property
    def f(self):
        return self._f

    @property
    def c(self):
        return self._c


# def human_eva_dataset(skeleton=None):
#     """Skeleton defaults to s20."""
#     dataset = _HumanEva1Dataset()
#     if skeleton is not None:
#         from skeleton import s14, s16, s20
#         if skeleton == s20:
#             return dataset
#         elif skeleton == s16:
#             return dataset.map_poses(s20.to_s16)
#         elif skeleton == s14:
#             return dataset.map_poses(s20.to_s14)
#         else:
#             raise Exception('Unrecognized skeleton, %s' % skeleton)
#     else:
#         return dataset


if __name__ == '__main__':
    import numpy as np
    from human_pose_util.skeleton import vis3d, vis2d
    from human_pose_util.transforms.np_impl import transform_frame
    from human_pose_util.transforms.np_impl import project
    from skeleton import s14
    # from skeleton import s16
    from skeleton import s20
    import matplotlib.pyplot as plt
    # dataset = human_eva_dataset(skeleton=s16)
    example = get_example(list(train_ids())[0]).map({
        k: s20.to_s14 for k in ['p2', 'p3c', 'p3w']})
    p3w, p3c, p2, r, t, f, c = example['p3w', 'p3c', 'p2', 'r', 't', 'f', 'c']
    p3w = p3w[0]
    p3c = p3c[0]
    p2 = p2[0]
    p3c2 = transform_frame(p3w, r, t)
    print(np.max(np.abs(p3c - p3c2)) / np.max(np.abs(p3c)))
    p22 = project(p3c, f, c)
    print(np.max(np.abs(p22 - p2)) / np.max(np.abs(p2)))
    skeleton = s14
    vis3d(skeleton, p3w)
    vis3d(skeleton, p3c)
    vis2d(skeleton, p2)
    plt.show()
