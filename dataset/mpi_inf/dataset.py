from __future__ import division
import os
import h5py
import numpy as np
from skeleton import converters
from human_pose_util.dataset.interface import Dataset
from human_pose_util.transforms.np_impl import euler_from_matrix_nh
from human_pose_util.transforms import np_impl

from human_pose_util.dataset.spec import scaled_dataset
from human_pose_util.dataset.spec import filter_by_camera
from human_pose_util.dataset.spec import modified_fps_dataset

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
            subjects.sort()
            for subject in subjects:
                print('Starting subject %s' % subject)
                subj_dir = os.path.join(dataset_dir, subject)
                subj_group = f.create_group(subject)
                sequences = [seq for seq in os.listdir(subj_dir)]
                sequences.sort()
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
        r = np.array(euler_from_matrix_nh(R))
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
            'camera_id': camera,
            'subject_id': subject,
            'sequence_id': sequence,
            'space_scale': 1,
            'pixel_scale': 1,
        }

    def __getitem__(self, key):
        if key == 'p3w':
            p3c = self['p3c']
            r = self.attrs['r']
            t = self.attrs['t']
            return np_impl.transform_frame(p3c, r, t, inverse=True)
        return self._converter.convert(
            self._group[MpiInfExample._group_map[key]][self._camera])

    @property
    def attrs(self):
        return self._attrs


def get_mpi_inf_dataset(
        train=True, skeleton_id='relevant', space_scale=1, pixel_scale=1,
        cameras=None, fps=None):
    """
    Get the `mpi_inf` dataset with the specified parameters.

    Args:
        train: train or eval specifier
        skeleton_id: one of ['relevant', 'base', 'extended']
        space_scale: scaling factor applied to 3D positions
        pixel_scale: scaling factor applied to 2D positions
        cameras: list of indices of cameras to use.
        fps: changes frame rate if specified.
    """
    dataset = MpiInfDataset(skeleton_id)
    if not train:
        raise NotImplementedError()
    if pixel_scale != 1 or space_scale != 1:
        dataset = scaled_dataset(
            pixel_scale=pixel_scale, space_scale=space_scale)
    if cameras is not None:
        dataset = filter_by_camera(dataset, cameras)
    if fps is not None:
        dataset = modified_fps_dataset(dataset, fps)
    return dataset


def register_all():
    from human_pose_util.serialization import dataset_register
    from human_pose_util.serialization import register_dataset_id_fn
    from human_pose_util.serialization import skeleton_register
    from skeleton import relevant, base, extended

    skeleton_register['mpi-inf-relevant'] = relevant
    skeleton_register['mpi-inf-base'] = base
    skeleton_register['mpi-inf-extended'] = extended

    dataset_register['mpi-inf'] = get_mpi_inf_dataset
    register_dataset_id_fn(
        'mpi-inf', os.path.join(_root_dir, 'default_datasets.json'))


def reorient_p3w(p3w):
    """Reorient p3w and shift for better visualization."""
    from human_pose_util.transforms.np_impl import rotate_about
    print('rotating')
    p3w = rotate_about(p3w, np.pi/2, 0)
    p3w[..., -1] -= np.min(p3w, axis=(0, 1))[-1]
    return p3w


if __name__ == '__main__':
    from skeleton import skeletons
    skeleton_id = 'relevant'
    dataset = MpiInfDataset(skeleton_id)
    skeleton = skeletons[skeleton_id]
    keys = list(dataset.keys())

    def check_world_coordinates():
        k0 = keys[0]
        k1 = keys[1]
        p3ws = [dataset[k]['p3w'] for k in [k0, k1]]
        print(np.max(np.abs(p3ws[0] - p3ws[1])))

    def check_height():
        example = dataset[keys]
        p3w = example['p3w']
        print(np.max(skeleton.height(p3w)))

    def check_coordinate_transforms():
        example = dataset[keys]
        p3w = example['p3w']
        p3c = example['p3c']
        r = example.attrs['r']
        t = example.attrs['t']
        transformed = np_impl.transform_frame(p3w, r, t)
        print(np.max(np.abs(transformed - p3c)), np.max(np.abs(p3c)))

    def check_projections():
        from human_pose_util.skeleton import vis2d
        import matplotlib.pyplot as plt
        example = dataset[keys[0]]
        p2 = example['p2']
        mins = np.min(p2, axis=1)
        maxs = np.max(p2, axis=1)
        n = len(mins)
        plt.plot(range(n), mins, range(n), maxs)
        plt.show()
        p3c = example['p3c']
        f = example.attrs['f']
        c = example.attrs['c']
        actual = np_impl.project(p3c, f, c)
        idx = -1
        plt.figure()
        vis2d(skeleton, actual[idx])
        plt.figure()
        vis2d(skeleton, p2[idx], linewidth=4)
        plt.show()
        print(np.max(np.abs(actual - p2)), np.max(np.abs(p2)))

    check_projections()

    # p3c = example['p3c']
    # p3w = example['p3w']
    # r = example.attrs['r']
    # R = example.attrs['R']
    # R_exp = np_impl.rotation_matrix(r)
    # t = example.attrs['t']

    # print(np.max(np.abs(p3c - p3w)))

    # p3c = np_impl.transform_frame(p3w, r, t, inverse=True)
    # p3c = reorient_p3w(p3c)

    # p3c_t = np_impl.transform_frame(p3w, r=r, t=t)
    # print(np.allclose(p3c, p3c_t))
    # print(np.max(np.abs(p3c - p3c_t)))
    # print(p3c[0] - p3c_t[0])

    # p3w_t = np_impl.transform_frame(p3c, r=r, t=t)
    # print(np.allclose(p3w, p3w_t))
    # print(np.max(np.abs(p3w - p3w_t)))
    # print(p3w[0] - p3w_t[0])

    # def proc(p3w, p3c):
    #     from human_pose_util.evaluate import procrustes_error
    #     print(np.max(np.abs(procrustes_error(p3w, p3c))))

    # proc(p3w, p3c)

    # def _vis_glumpy(p0, p1=None):
    #     from skeleton import relevant
    #     from human_pose_util.animation.animated_scene import \
    #         add_limb_collection_animator
    #     from human_pose_util.animation.animated_scene import run
    #     skeleton = relevant
    #     fps = dataset.attrs['fps']
    #     if p0 is not None:
    #         add_limb_collection_animator(
    #             skeleton, p0 / 2000, fps, linewidth=2)
    #     if p1 is not None:
    #         add_limb_collection_animator(
    #             skeleton, p1 / 2000, fps, linewidth=4)
    #     run(fps=dataset.attrs['fps'])
    #
    # # print(np.min(p3w))
    # # print(np.max(p3w))
    # # print(np.max(np.abs(p3c - p3w)))
    # # _vis_glumpy(p3c, p3w)
    # p3w = reorient_p3w(p3ws[0])
    # print(skeleton.front_angle(p3w).shape)
    # p3w_r = np_impl.rotate_about(
    #     p3w, -np.expand_dims(skeleton.front_angle(p3w), axis=-1), 2)
    # root = skeleton.joint_index(skeleton.root_joint)
    # p3w_r[..., :2] -= p3w_r[..., root:root+1, :2]
    # _vis_glumpy(p3w, p3w_r)
