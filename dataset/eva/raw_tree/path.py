"""Provides convenience functions for locating paths."""
import os
import string

try:
    uncompressed_dir = os.environ['HUMAN_EVA_1_PATH']
except KeyError:
    raise KeyError('Environment variable HUMAN_EVA_1_PATH not set.')


class ExistingPathGetter:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        path = self.fn(*args, **kwargs)
        if not os.path.exists(path):
            raise IOError('Path %s does not exist' % path)
        return path


@ExistingPathGetter
def mocap_dir(subject):
    return os.path.join(uncompressed_dir, subject, 'Mocap_Data')


@ExistingPathGetter
def mocap_data_path(subject, sequence):
    return _mocap_data_path(subject, sequence)


def _mocap_data_path(subject, sequence):
    return os.path.join(mocap_dir(subject), '%s.mat' % sequence)


def is_sequence(subject, sequence):
    return os.path.isfile(_mocap_data_path(subject, sequence))


def sequences(subject, base_sequence=None):
    sequences = [string.join(s.split('_')[:2], '_')
                 for s in os.listdir(video_dir(subject))]
    if base_sequence is not None:
        if isinstance(base_sequence, str):
            sequences = [s for s in sequences if s.startswith(base_sequence)]
        else:
            sequences = [s for s in sequences if
                         s.split('/')[0] in base_sequence]
    sequences = [s for s in set(sequences) if is_sequence(subject, s)]
    sequences.sort()
    return sequences


@ExistingPathGetter
def camera_cal_path(subject, camera):
    return os.path.join(
        uncompressed_dir, subject, 'Calibration_DATA', '%s.cal' % camera)


@ExistingPathGetter
def video_dir(subject):
    return os.path.join(uncompressed_dir, subject, 'Image_Data')


def _video_path(subject, sequence, camera):
    return os.path.join(video_dir(subject), '%s_(%s).avi' % (sequence, camera))


@ExistingPathGetter
def video_path(subject, sequence, camera):
    return _video_path(subject, sequence, camera)


def is_view(subject, sequence, camera):
    return os.path.isfile(_video_path(subject, sequence, camera))


@ExistingPathGetter
def cal_path(subject, camera):
    return os.path.join(
        uncompressed_dir, subject, 'Calibration_Data', '%s.cal' % camera)


@ExistingPathGetter
def mp_path(subject):
    return os.path.join(mocap_dir(subject), '%s.mp' % subject)


@ExistingPathGetter
def ofs_dir(subject):
    return os.path.join(uncompressed_dir, subject, 'Sync_Data')


@ExistingPathGetter
def ofs_path(subject, sequence, camera):
    return os.path.join(ofs_dir(subject), '%s_(%s).ofs' % (sequence, camera))
