from __future__ import division
import os
import h5py
import numpy as np
from skeleton import converters
from human_pose_util.dataset.interface import Dataset
from human_pose_util.transforms.np_impl import euler_from_matrix_nh


n_cameras = 14


def _parse_calibration_file(path):
    import re
    if not os.path.isfile(path):
        raise ValueError('No file at path: %s' % path)
    keys = ['sensor', 'size', 'animated', 'intrinsic', 'extrinsic', 'radial']
    with open(path, 'r') as f:
        line = f.readline().strip()
        assert(line == 'Skeletool Camera Calibration File V1.0')
        ws = re.compile('[ \t]*')
        line = f.readline().strip()
        count = 0
        all_vals = []
        while line:
            assert(line[:4] == 'name')
            name = re.split(ws, line)[-1]
            assert(int(name) == count)
            count += 1
            vals = []
            for key in keys:
                entries = re.split(ws, f.readline().strip())
                assert(entries[0] == key)
                vals.append([float(e) for e in entries[1:]])
            all_vals.append(vals)
            line = f.readline().strip()
    return keys, all_vals


def get_dataset_dir():
    """
    Get the dataset directory based on MPI_INF_PATH environment variable.

    Raises:
        RuntimeError if MPI_INF_PATH environment not set or isn't a directory.
    """
    if 'MPI_INF_PATH' in os.environ:
        dataset_dir = os.environ['MPI_INF_PATH']
        if not os.path.isdir(dataset_dir):
            raise RuntimeError('MPI_INF_PATH directory does not exist')
        return dataset_dir
    else:
        raise RuntimeError('MPI_INF_PATH environment variable not set.')


_root_dir = os.path.realpath(os.path.dirname(__file__))


_data_path = os.path.join(_root_dir, 'data.hdf5')


def to_hdf5(overwrite=False):
    """Convert annotations to hdf5."""
    import scipy.io
    if not overwrite and os.path.isfile(_data_path):
        raise RuntimeError('hdf5 file already exists %s. '
                           'Use overwrite if sure, or delete the file.')
    dataset_dir = get_dataset_dir()
    try:
        with h5py.File(_data_path, 'w') as f:
            subjects = [s for s in os.listdir(dataset_dir) if s[0] == 'S']
            for subject in subjects:
                print('Starting subject %s' % subject)
                subj_dir = os.path.join(dataset_dir, subject)
                subj_group = f.create_group(subject)
                sequences = [seq for seq in os.listdir(subj_dir)]
                for sequence in sequences:
                    seq_dir = os.path.join(subj_dir, sequence)
                    seq_group = subj_group.create_group(sequence)
                    annot_path = os.path.join(seq_dir, 'annot.mat')
                    annot_data = scipy.io.loadmat(annot_path)
                    for k in ['univ_annot3', 'annot3', 'annot2']:
                        n = int(k[-1])
                        data = annot_data[k]
                        data = np.array([d[0] for d in data], dtype=np.float32)
                        shape = data.shape
                        data = np.reshape(
                            data, (shape[0], -1, shape[2] // n, n))
                        seq_group.create_dataset(k, data=data)
                    camera_path = os.path.join(seq_dir, 'camera.calibration')
                    keys, camera_data = _parse_calibration_file(camera_path)
                    camera_data = zip(*camera_data)
                    camera_data_dict = {
                        k: v for k, v in zip(keys, camera_data)}
                    seq_group.attrs['intrinsic'] = np.reshape(np.array(
                        camera_data_dict['intrinsic'], dtype=np.float32),
                        (-1, 4, 4))[:, :2, :3]

                    seq_group.attrs['extrinsic'] = np.reshape(np.array(
                        camera_data_dict['extrinsic']), (-1, 4, 4))[:, :3, :]

                    seq_group.attrs['shape'] = np.array(
                        camera_data_dict['size'], dtype=np.int32)

                    seq_group.attrs['folder'] = seq_dir

                    print('Finished %s_%s' % (subject, sequence))

    except Exception:
        print('Error occured: deleting hdf5 data file.')
        os.remove(_data_path)
        raise


if not os.path.isfile(_data_path):
    to_hdf5()


class MpiInfDataset(Dataset):

    _root_group = h5py.File(_data_path, 'r')

    def __init__(self, skeleton_id):
        self._converter = converters[skeleton_id]
        self._attrs = {
            'skeleton_id': skeleton_id,
            'fps': 25,
        }

    def keys(self):
        subjects = list(MpiInfDataset._root_group)
        subjects.sort()
        for subject in subjects:
            s_group = MpiInfDataset._root_group[subject]
            sequences = list(s_group)
            sequences.sort()
            for sequence in sequences:
                for camera in range(n_cameras):
                    yield '%s_%s_%d' % (subject, sequence, camera)

    def __getitem__(self, key):
        subj, seq, camera = key.split('_')
        return MpiInfExample(subj, seq, int(camera), self._converter)

    @property
    def attrs(self):
        return self._attrs


class MpiInfExample(Dataset):
    _group_map = {
        'p3w': 'univ_annot3',
        'p3c': 'annot3',
        'p2': 'annot2',
    }

    _keys = frozenset(['p3w', 'p3c', 'p2'])

    def __init__(self, subject, sequence, camera, converter):
        if not (isinstance(camera, int) and 0 <= camera < n_cameras):
            raise ValueError(
                'camera must be an int in [0, %d), got %s'
                % (n_cameras, camera))
        self._camera = camera
        self._group = MpiInfDataset._root_group[subject][sequence]
        extr = self._group.attrs['extrinsic'][camera]
        R = extr[:, :3]
        t = extr[:, 3]
        intr = self._group.attrs['intrinsic'][camera]
        f = np.diag(intr)
        c = intr[:, 2]
        r = euler_from_matrix_nh(R)
        shape = self._group.attrs['shape'][camera]
        self._converter = converter
        self._attrs = {
            'r': r,
            't': t,
            'R': R,
            'f': f,
            'c': c,
            'fps': 25,
            'shape': shape,
            'camera': camera
        }

    def __getitem__(self, key):
        return self._converter.convert(
            self._group[MpiInfExample._group_map[key]][self._camera])

    @property
    def attrs(self):
        return self._attrs


def reorient_p3w(p3w):
    """Reorient p3w and shift for better visualization."""
    from human_pose_util.transforms.np_impl import rotate_about
    print('rotating')
    p3w = rotate_about(p3w, -np.pi/2, 0)
    p3w[..., -1] -= np.min(p3w, axis=(0, 1))[-1]
    return p3w


if __name__ == '__main__':
    from human_pose_util.animation.animated_scene import \
        add_limb_collection_animator
    from human_pose_util.animation.animated_scene import run
    from skeleton import relevant
    dataset = MpiInfDataset('relevant')
    key = list(dataset.keys())[0]
    example = dataset[key]

    p3w = reorient_p3w(example['p3w']) / 2000
    skeleton = relevant
    add_limb_collection_animator(skeleton, p3w, dataset.attrs['fps'])
    run(fps=dataset.attrs['fps'])
