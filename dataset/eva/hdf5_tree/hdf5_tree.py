"""Script for converting downloaded data to hdf5 archive."""
from __future__ import division
import os
import h5py
import numpy as np
import human_eva.data_tree as data_tree

data_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data.hdf5')


def data_file(mode='r'):
    return h5py.File(data_path, mode)


def data_file_exists():
    return os.path.isfile(data_path)


class Hdf5Node(object):
    def __init__(self, group):
        self._group = group


class RootNode(Hdf5Node, data_tree.RootNode):

    def subject(self, subject_code):
        subjects_group = self._subjects_group
        if subject_code not in subjects_group:
            raise Exception('Invalid subject_code %s.' % subject_code)
        return SubjectNode(subjects_group[subject_code])

    @property
    def subject_codes(self):
        return [k for k in self._subjects_group]

    @property
    def _subjects_group(self):
        return self._group['subjects']


class SubjectNode(Hdf5Node, data_tree.SubjectNode):
    def sequence(self, sequence_code):
        if sequence_code not in self._sequences_group:
            raise Exception('Invalid sequence_code: %s' % sequence_code)
        return SequenceNode(self._sequences_group[sequence_code])

    @property
    def parent(self):
        return RootNode(self._group.parent.parent)

    @property
    def subject_code(self):
        return self._group.name.split('/')[-1]

    def sequence_codes(self, base_code=None):
        codes = [k for k in self._sequences_group]
        if base_code is not None:
            if isinstance(base_code, str):
                codes = [c for c in codes if c.startswith(base_code)]
            else:
                codes = [c for c in codes if c.split('_')[0] in base_code]
        return codes

    @property
    def _sequences_group(self):
        return self._group['sequences']


class SequenceNode(Hdf5Node, data_tree.SequenceNode):
    @property
    def subject(self):
        return SubjectNode(self._group.parent.parent)

    @property
    def sequence_code(self):
        return self._group.name.split('/')[-1]

    def view(self, camera_code):
        if camera_code not in self._views_group:
            raise Exception('Invalid camera_code: %s' % camera_code)
        return ViewNode(self._views_group[camera_code])

    @property
    def camera_codes(self):
        return [k for k in self._views_group]

    @property
    def _views_group(self):
        return self._group['views']

    # @property
    # def fps(self):
    #     return self._group.attrs['fps']


class ViewNode(Hdf5Node, data_tree.ViewNode):
    @property
    def sequence(self):
        return SequenceNode(self._group.parent.parent)

    @property
    def camera_code(self):
        return self._group.name.split('/')[-1]

    @property
    def n_frames(self):
        return self._group.attrs['n_frames']

    @property
    def p3_camera(self):
        return self._data('p3_camera')

    @property
    def p3_world(self):
        return self._data('p3_world')

    @property
    def p2(self):
        return self._data('p2')

    @property
    def camera_extrinsics(self):
        return self._data('r'), self._data('t')

    @property
    def camera_intrinsics(self):
        return self._data('f'), self._data('c')

    def _data(self, key):
        return np.array(self._group[key])

    # def _convert(self):
    #     if not os.path.isfile(self.video_path):
    #         raise Exception('No video at %s' % self.video_path)
    #     print(self.subject_code, self.sequence_code, self.camera_code)


def _converted():
    if not os.path.isfile(data_path):
        return False
    with h5py.File(data_path, 'r') as f:
        ret = 'subjects' in f
    return ret


def convert():
    from human_pose_util.dataset.eva.raw_tree import root_node

    def copy_data(view, views_group):
        try:
            sequence = view.sequence
            group = views_group.require_group(view.code)
            if sequence.has_joint_data:
                if 'p3_camera' not in group:
                    group.create_dataset('p3_camera', data=view.p3_camera)
                if 'p3_world' not in group:
                    group.create_dataset('p3_world', data=view.p3_world)
                if 'p2' not in group:
                    group.create_dataset('p2', data=view.p2)
            r, t = view.camera_extrinsics
            if 'r' not in group:
                group.create_dataset('r', data=r)
            if 't' not in group:
                group.create_dataset('t', data=t)
            f, c = view.camera_intrinsics
            if 'f' not in group:
                group.create_dataset('f', data=f)
            if 'c' not in group:
                group.create_dataset('c', data=c)
            print('Successfully converted native -> hdf5: %s' % str(view))
        except RuntimeError:
            print('Failed to convert: %s' % view)
            del views_group[view.code]
            # raise

    with h5py.File(data_path, 'a') as f:
        subjects_group = f.require_group('subjects')

        for subject in root_node.subjects:
            subject_group = subjects_group.require_group(subject.code)
            sequences_group = subject_group.require_group('sequences')
            for sequence in subject.sequences():
                sequence_group = sequences_group.require_group(
                    sequence.code)
                views_group = sequence_group.require_group('views')
                for view in sequence.views:
                    copy_data(view, views_group)


def _print_views():
    with h5py.File(data_path, 'r') as f:
        rn = RootNode(f)
        for subject in rn.subjects:
            for sequence in subject.sequences():
                for view in sequence.views:
                    print(subject.code, sequence.code, view.code)
                    if sequence.has_joint_data:
                        print(view.p3_world.shape)


def _clean():
    with h5py.File(data_path, 'a') as f:
        rn = RootNode(f)
        for subject in rn.subjects:
            for sequence in subject.sequences():
                views_group = sequence._group['views']
                for view in sequence.views:
                    print(subject.code, sequence.code, view.code)
                    continue
                    # print(view)
                    if 'p3_camera' not in view._group:
                        print('No p3_camera data. Removing. %s' % str(view))
                        del views_group[view.code]
                    else:
                        p3 = view._group['p3_camera']
                        invalid = np.sum(np.isnan(p3)) / np.sum(
                                np.ones_like(p3))
                        if invalid > 0.1:
                            print('Too many nans. Removing. %s' % str(view))
                            del views_group[view.code]


# def _print_native():
#     from native_tree import root_node
#     rn = root_node
#     for subject in rn.subjects:
#         for sequence in subject.sequences():
#             for view in sequence.views:
#                 print(subject.code, sequence.code, view.code)


if __name__ == '__main__':
    # if not _converted():
    # _convert()
    # _clean()
    _print_views()
    # _print_native()
