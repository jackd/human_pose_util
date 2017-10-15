"""Provides functionality for reading pose data/filenames from h5py file."""
import string
import os
import h5py
import numpy as np
from itertools import chain
# import string

root_dir = os.path.dirname(os.path.realpath(__file__))
# dataset_dir = os.path.realpath(os.path.join(root_dir, 'data', 'data'))
# _videos_dir = os.path.join(dataset_dir, 'Videos')

_train_subjects = set(['S1', 'S5', 'S6', 'S7', 'S8'])
_eval_subjects = set(['S9', 'S11'])
_test_subjects = set(['S2', 'S3', 'S4'])


def get_dataset_dir():
    if 'H3M_PATH' in os.environ:
        dataset_dir = os.environ['H3M_PATH']
        if not os.path.isdir(dataset_dir):
            raise Exception('H3M_PATH directory does not exist')
        return dataset_dir
    else:
        raise Exception('H3M_PATH environment variable not set.')


def _train_subject_codes(self):
    """Get an iterable of subject codes for training."""
    for s in _train_subjects:
        yield s


def _eval_subject_codes(self):
    """Get an iterable of subject codes for evaluation."""
    for s in _eval_subjects:
        yield s


def _subject_codes(include_train=True, include_eval=True, include_test=False):
    """Get an iterable of subject codes in the included sets."""
    sets = []
    if include_train:
        sets.append(_train_subjects)
    if include_eval:
        sets.append(_eval_subjects)
    if include_test:
        sets.append(_test_subjects)
    return chain(*sets)


_hdf5_path = os.path.join(root_dir, 'data.hdf5')


def has_hdf5_data():
    return os.path.isfile(_hdf5_path)


# def chain(iterables):
#     for iterable in iterables:
#         for value in iterable:
#             yield value


def convert():
    from spacepy import pycdf
    import imageio
    from h3m.skeleton import original_limb_indices
    print('Converting h3m data from cdf to hdf5')
    dataset_dir = get_dataset_dir()

    # from h3m.skeleton import skeleton
    # from h3m.transforms import normalize, abs_to_rel

    def _p2_path(subject, sequence, view):
        return os.path.join(
            dataset_dir, 'D2_positions', subject,
            '%s.%s.cdf' % (sequence, view))

    def _p3_path(subject, sequence, view=None):
        if view is None:
            return os.path.join(
                dataset_dir, 'D3_positions', subject, '%s.cdf' % sequence)
        else:
            return os.path.join(
                dataset_dir, 'D3_positions_mono', subject,
                '%s.%s.cdf' % (sequence, view))

    def _video_path(subject, sequence, view):
        return os.path.join(
            dataset_dir, 'Videos', subject, '%s.%s.mp4' % (sequence, view))

    def _sequences(subject):
        # return (p[:-4] for p in
        #         os.listdir(os.path.join(dataset_dir, 'Videos', subject))
        #         if p[:4] != '_ALL')
        return (p[:-4] for p in os.listdir(
            os.path.join(dataset_dir, 'D3_positions', subject)))

    def _views(subject, sequence):
        paths = os.listdir(os.path.join(dataset_dir, 'Videos', subject))
        n = len(sequence)
        return (p[n+1:-4] for p in paths if
                len(p) > n + 1 and p[:n+1] == '%s.' % sequence)

    _filter_indices = original_limb_indices()
    train_eval_subjects = _train_subjects.union(_eval_subjects)

    def _convert_subject(subject, group):
        sequences_group = group.create_group('sequences')
        for sequence in _sequences(subject):
            sequence_group = sequences_group.create_group(sequence)
            s = _convert_sequence(subject, sequence, sequence_group)
            if not s:
                del sequences_group[sequence]

    def _convert_sequence(subject, sequence, group):
        print('Converting %s...' % group.name)
        views = list(_views(subject, sequence))
        n_video_frames = None
        for view in views:
            # print(subject, sequence, view)
            try:
                video_path = _video_path(subject, sequence, view)
                with imageio.get_reader(video_path) as reader:
                    n = len(reader)
                    n_video_frames = n if n_video_frames is None else \
                        min(n_video_frames, n)
            except Exception:
                print('Failed to read video file. Skipping...')
                return False

        n = n_video_frames

        with pycdf.CDF(_p3_path(subject, sequence)) as cdf:
            p3 = cdf['Pose'][0]
            n_frames = p3.shape[0]
            n = min(n_frames, n)
            p3 = p3.reshape(n_frames, -1, 3)[:n, _filter_indices]
            p3d = group.create_dataset('p3', p3.shape, dtype=np.int32)
            p3d[...] = p3

        group.attrs['len'] = n

        # p3 = normalize(np.array(p3d, dtype=np.float32), subject)
        # theta = group.create_dataset('theta', (n,), dtype=np.float32)
        # rel_poses = group.create_dataset(
        #     'rel_p3', (n, skeleton.n_joints, 3), dtype=np.float32)
        # for i, abs_pose in enumerate(p3):
        #     rel_poses[i], theta[i] = abs_to_rel(abs_pose)

        views_group = group.create_group('views')
        for view in views:
            view_group = views_group.create_group(view)
            with pycdf.CDF(_p2_path(subject, sequence, view)) as cdf:
                p2 = cdf['Pose'][0]
                assert(p2.shape[0] == n_frames)
                p2 = p2.reshape(n_frames, -1, 2)[:n, _filter_indices]
                p2d = view_group.create_dataset('p2', p2.shape, dtype=np.int32)
                p2d[...] = p2

            with pycdf.CDF(_p3_path(subject, sequence, view)) as cdf:
                p3 = cdf['Pose'][0]
                assert(p3.shape[0] == n_frames)
                p3 = p3.reshape(n_frames, -1, 3)[:n, _filter_indices]
                p3d = view_group.create_dataset('p3', p3.shape, dtype=np.int32)
                p3d[...] = p3

        return True

    def _convert_all(dataset):
        dataset.attrs['n_joints'] = len(_filter_indices)
        subject_group = dataset.create_group('subjects')
        for subject in train_eval_subjects:
            group = subject_group.create_group(subject)
            _convert_subject(subject, group)

    with h5py.File(_hdf5_path, 'w') as f:
        _convert_all(f)


class RootNode(object):
    """Class for wrapping h5py file."""

    def __init__(self, mode='r'):
        """Initialize with read/write mode, e.g. 'r', 'w', 'a'."""
        self._mode = mode
        if not has_hdf5_data():
            convert()

    def __enter__(self):
        """Open the stored h5py file for reading."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the h5py file this is based on."""
        self.exit()

    def open(self):
        """Open the backing file."""
        self._file = h5py.File(_hdf5_path, self._mode)

    def exit(self):
        """Close the backing file."""
        self._file.close()

    def subjects(self, include_train=True, include_eval=True,
                 include_test=False):
        """Get an iterable of subjects stored here."""
        subjects = _subject_codes(
            include_train=include_train,
            include_eval=include_eval,
            include_test=include_test)
        return (self.subject(s) for s in subjects)

    def subject(self, subject_code):
        """Get the subject with the given code."""
        if isinstance(subject_code, int):
            subject_code = 'S%d' % subject_code
        return SubjectNode(self._file['subjects/%s' % subject_code])

    def sequences(self, include_train=True, include_eval=True,
                  include_test=False):
        """Get an iterable of all sequences."""
        return chain(*(s.sequences for s in self.subjects(
            include_train=include_train,
            include_eval=include_eval,
            include_test=include_test
        )))

    def views(self, include_train=True, include_eval=True, include_test=False):
        """Get an iterable of all views."""
        return chain(*(s.views for s in self.subjects(
            include_train=include_train,
            include_eval=include_eval,
            include_test=include_test
        )))

    def view(self, view_path):
        return ViewNode(self._file[view_path])

    @property
    def n_joints(self):
        """The number of annotated joints in this dataset."""
        return self._file.attrs['n_joints']


class SubjectNode(object):
    """Class representing all examples from a single subject."""

    def __init__(self, group):
        """Initialize with the h5py group with data stored for this subject."""
        self._group = group

    @property
    def sequences(self):
        """Get an iterable of sequences stored for this subject."""
        sequences_group = self._group['sequences']
        return (self.sequence(s) for s in sequences_group)

    def sequence(self, sequence_code):
        """Get the sequence with the associated code."""
        if isinstance(sequence_code, int):
            sequence_code = 'S%d' % sequence_code
        return SequenceNode(self._group['sequences/%s' % sequence_code])

    @property
    def views(self):
        """Get an iterable of views stored for this subject."""
        return chain(*(s.views for s in self.sequences))

    def __str__(self):
        """String representation."""
        return self._group.name[1:]

    def __repr__(self):
        """String representation."""
        return self._group.name[1:]


class SequenceNode(object):
    """Class representing all records from a single sequence."""

    def __init__(self, group):
        """Initialize with the h5py group with data stored for the sequence."""
        self._group = group

    def __len__(self):
        """Number of records for this sequence."""
        return self._group.attrs['len']

    def view(self, view_code):
        """Get the view for the given camera code."""
        if isinstance(view_code, int):
            view_code = _view_codes[view_code]
        return ViewNode(self._group['views'][view_code])

    @property
    def views(self):
        """Get the views for this sequence."""
        return (self.view(s) for s in self._group['views'])

    def has_view(self, view_code):
        """Bool indicating presence of bool."""
        return view_code in self._group['views']

    @property
    def p3(self):
        """Get the 3d pose data for this sequence."""
        return np.array(self._group['p3'], dtype=np.float32)

    def __str__(self):
        """String representation."""
        return self._group.name[1:]

    def __repr__(self):
        """String representation."""
        return self._group.name[1:]

    @property
    def subject(self):
        """Get the associated subject."""
        return SubjectNode(self._group.parent.parent)


def _view_index(view):
    if isinstance(view, ViewNode):
        view = view.index
    elif isinstance(view, str):
        view = _view_index[view]
    return view


# def view_angles(view):
#     """Get the angles for the given view. Can be ViewNode, index or code."""
#     # elif not isinstance(view, int):
#     #     raise TypeError('view must be ViewNode, int of str')
#     return _view_angles[_view_index(view)]
#
#
# def view_offset(view):
#     """
#     Get the offset in world coordinates of the given view.
#
#     `view` can be ViewNode, index or code.
#     """
#     return _view_offsets[_view_index(view)]


# Dirty hardcoded values from view_transforms.py
# _view_angles = [
#     [1.38183773, -3.1033566, -0.43632239],
#     [1.35548615, 3.15477443, -2.7685194],
#     [1.36247396, -3.21094561, 0.40532005],
#     [-1.83953595, -0.06365179, -0.40930146],
# ]
_view_codes = ['54138969', '55011271', '58860488', '60457274']

# world-to-camera transform params,
# i.e. change_frame(sequence.p3, **_transform_params[view.code]) = view.p3
# _transform_params = {
#     '54138969': {
#         'ai': 1.3666989415052759,
#         'aj': 2.7307938783659531,
#         'ak': -0.056486023929749329,
#         'offset': [-345.20440656, 546.75926902, 5473.67261914]
#     },
#     '55011271': {
#         'ai': 1.8027789117649644,
#         'aj': 0.37138590004215577,
#         'ak': 0.087942362601729246,
#         'offset': [250.86954742, 420.60100609, 5588.31752961]
#     },
#     '58860488': {
#         'ai': 1.3345080330215466,
#         'aj': -2.7270441069039881,
#         'ak': 0.0498403120038634,
#         'offset': [480.1449599, 253.6888976, 5703.11348413]
#     },
#     '60457274': {
#         'ai': 1.8769247254788062,
#         'aj': -0.41310978122820874,
#         'ak': -0.06174031493127341,
#         'offset': [51.73843346, 378.06763716, 4405.94346249]
#     }
# }
#
# _view_angles = np.array(
#     [[_transform_params[k][i]
#         for i in ['ai', 'aj', 'ak']]
#         for k in _view_codes])
#
#
# _view_offsets = np.array([_transform_params[k] for k in _view_codes])


def get_view_codes():
    return list(_view_codes)


def view_index(view_code):
    return _view_index[view_code]


# _view_index = {
#     k: i for i, k in enumerate(_view_codes)
# }


class ViewNode(object):
    """Class representing all records for a single view of a sequence."""

    def __init__(self, group):
        """Initialize with the h5py group with data stored for the view."""
        self._group = group

    @property
    def p2(self):
        """Get the 2d pose data for this view."""
        return np.array(self._group['p2'])

    @property
    def p3(self):
        """Get the 3d pose data for this view."""
        return np.array(self._group['p3'])

    @property
    def video_path_short(self):
        """Get the short path, e.g. `S1/Eating blah.blah.mp4`."""
        s = string.split(self._group.name, '/')
        subject_string = s[2]
        sequence_string = s[4]
        view_string = s[6]
        return os.path.join(
            subject_string, '%s.%s.mp4' % (sequence_string, view_string))

    @property
    def video_path(self):
        """Get the full path to the video for this view."""
        videos_dir = os.path.join(get_dataset_dir(), 'Videos')
        return os.path.join(videos_dir, self.video_path_short)

    def __str__(self):
        """String representation."""
        return self._group.name[1:]

    def __repr__(self):
        """String representation."""
        return self._group.name[1:]

    def __len__(self):
        """Number of records for the sequence corresponding to this view."""
        return self._group.parent.attrs['len']

    @property
    def code(self):
        """Get the camera code associated with this view."""
        return str(self).split('/')[-1]

    @property
    def index(self):
        """Get this view's index."""
        return _view_index[self.code]

    # @property
    # def angles(self):
    #     """Get the view angle for this view compared to it's sequence."""
    #     return view_angles(self)

    @property
    def sequence(self):
        """Get the sequence associated with this view."""
        return SequenceNode(self._group.parent.parent)

    # def verify(self):
    #     """Verify the video is readable."""
    #     n = len(self)
    #     with imageio.get_reader(self.video_path) as reader:
    #         print(n)
    #         print(len(reader))


def _root_node(mode):
    return RootNode(mode)


def root_node():
    """Get the root node for all other data."""
    return _root_node('r')


if __name__ == '__main__':
    if not os.path.exists(_hdf5_path):
        try:
            convert()
        except Exception:
            if os.path.isfile(_hdf5_path):
                os.remove(_hdf5_path)
            raise
    else:
        import imageio
        from skeleton import s24
        from human_pose_util.skeleton import vis3d, vis2d, plt
        with root_node() as rn:
            for sequence in rn.sequences():
                vis3d(s24, sequence.p3[0])
                plt.show()
                for view in sequence.views:
                    with imageio.get_reader(view.video_path) as reader:
                        # frame = reader.get_data(0)
                        frame = reader.get_next_data()
                    p2 = view.p2[0]
                    p3 = view.p3[0]
                    ax = plt.gca()
                    ax.imshow(frame)
                    vis2d(s24, p2, ax=ax, change_ax=False)
                    plt.show()
