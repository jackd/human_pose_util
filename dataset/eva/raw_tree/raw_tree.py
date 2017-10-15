from __future__ import division

import functools
import collections
import numpy as np
import cv2
from human_pose_util.dataset.eva.skeleton import s20
from human_eva.path import mocap_data_path, cal_path
from camera_parameter import CameraParameter
from actor_parameter import ActorParameter
from motion_capture_data import MotionCaptureData
from conic_limb_parameter import ConicLimbParameter
from limb_length import LimbLength
from global_marker_transform import GlobalMarkerTransform
from synchronization_parameter import SynchronizationParameter
from pose_2d import Pose2D
from pose_3d import Pose3D
from human_eva.data_tree import RootNode, SubjectNode, SequenceNode, ViewNode
from human_eva.path import mp_path, ofs_path
from human_pose_util.dataset.eva.meta import subjects, cameras, base_sequences
_subjects_set = set(subjects)
_cameras_set = set(cameras)
_base_sequences_set = set(base_sequences)


def _change_frame(x, r, t):
    from transformations import euler_matrix
    if r.shape != (3,):
        raise Exception('Invalid shape for r: %s' % str(r.shape))
    if t.shape[-1] != 3:
        raise Exception('Invalid shape for t: %s' % str(t.shape))
    assert(len(t) == 3)
    R = euler_matrix(r[0], r[1], r[2])[:3, :3]
    return np.matmul(x, R.T) + t


def _projection(x, f=None, c=None):
    if (x.shape[-1] != 3):
        raise Exception('Invalid shape for x: %s' % str(x.shape))
    if not (f.shape == () or f.shape == 2,):
        raise Exception('Invalid shape for f: %s' % str(f.shape))
    if not (c.shape == () or c.shape == 2,):
        raise Exception('Invalid shape for c: %s' % str(c.shape))
    x = x[..., :2] / x[..., 2:3]
    if f is not None:
        x *= f
    if c is not None:
        x += c
    return x


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """

    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


class _RootNode(RootNode):
    @memoized
    def subject(self, subject_code):
        return _SubjectNode(subject_code)

    @property
    def subject_codes(self):
        return subjects


root_node = _RootNode()


class _SubjectNode(SubjectNode):
    def __init__(self, subject_code):
        if subject_code not in _subjects_set:
            raise Exception('subject_code %s not valid.' % subject_code)
        self._subject_code = subject_code

    @property
    def parent(self):
        return root_node

    @memoized
    def sequence(self, sequence_code):
        return _SequenceNode(self, sequence_code)

    @property
    def subject_code(self):
        return self._subject_code

    def sequence_codes(self, base_code=None):
        return ['%s_%d' % (b, i+1) for i in range(3) for b in base_sequences]

    @cached_property
    def actor_params(self):
        return ActorParameter(mp_path(self.code))


class _SequenceNode(SequenceNode):
    def __init__(self, subject, sequence_code):
        self._subject = subject
        self._sequence_code = sequence_code
        if self.base_sequence_code not in _base_sequences_set or not \
                0 <= self.trial_index < 3:
            raise Exception('invalid sequence_code %s' % sequence_code)

    @property
    def subject(self):
        return self._subject

    @memoized
    def view(self, camera_code):
        return _ViewNode(self, camera_code)

    @property
    def sequence_code(self):
        return self._sequence_code

    @property
    def camera_codes(self):
        return cameras
        # return [c for c in cameras if is_view(
        #     self.subject_code, self.sequence_code, c)]

    @property
    def code(self):
        return self._sequence_code

    @property
    def mocap_data_path(self):
        return mocap_data_path(self.subject_code, self.sequence_code)

    @cached_property
    def mocap_data(self):
        return MotionCaptureData(self.mocap_data_path)

    @cached_property
    def conic_limb_parameters(self):
        return ConicLimbParameter(
            self.mocap_data, self.subject.actor_params)

    def limb_length(self, mocap_frame):
        return LimbLength(self.mocap_data, mocap_frame)

    def global_marker_transform(self, mocap_frame):
        return GlobalMarkerTransform(
            self.mocap_data, mocap_frame, self.conic_limb_parameters,
            self.limb_length(mocap_frame))

    def p3_frame(self, mocap_frame):
        return Pose3D(
            self.global_marker_transform(mocap_frame),
            self.limb_length(mocap_frame))


class _ViewNode(ViewNode):
    def __init__(self, sequence, camera_code):
        self._sequence = sequence
        self._camera_code = camera_code

    @property
    def subject(self):
        return self._sequence.subject

    @property
    def sequence(self):
        return self._sequence

    @property
    def camera_code(self):
        return self._camera_code

    @property
    def code(self):
        return self._camera_code

    @cached_property
    def camera_params(self):
        return CameraParameter(cal_path(self.subject_code, self.camera_code))

    def mocap_frame(self, image_frame):
        if self.sequence.has_video:
            sync_param = self.sync_params
            return sync_param.mc_st + (
                image_frame - sync_param.im_st) * sync_param.mc_sc
        else:
            return image_frame

    def pose2d(self, image_frame):
        mocap_frame = self.mocap_frame(image_frame)
        return Pose2D(
            self.sequence.p3_frame(mocap_frame), self.camera_params)

    @cached_property
    def poses2d(self):
        return [self.pose2d(i) for i in range(self.n_frames)]

    @cached_property
    def sync_params(self):
        return SynchronizationParameter(
            ofs_path(self.subject_code, self.sequence_code, self.camera_code))

    @cached_property
    def video_n_frames(self):
        return int(self._get_from_video(cv2.CAP_PROP_FRAME_COUNT))

    @cached_property
    def video_fps(self):
        return self._get_from_video(cv2.CAP_PROP_FPS)

    def _get_from_video(self, code):
        cap = cv2.VideoCapture(self.video_path)
        val = cap.get(code)
        cap.release()
        return val

    def pose3d(self, image_frame):
        mocap_frame = self.mocap_frame(image_frame)
        return self.sequence.p3_frame(mocap_frame)

    @cached_property
    def poses3d(self):
        return [self.pose3d(i) for i in range(self.n_frames)]

    @cached_property
    def p3_world(self):
        return np.array([s20.from_raw(p) for p in self.poses3d])

    @cached_property
    def p3_camera(self):
        r, t = self.camera_extrinsics

        def f(x):
            return _change_frame(x, r=r, t=t)

        return _apply_to_flattened(self.p3_world, f)

    @cached_property
    def p2(self):
        f, c = self.camera_intrinsics
        return _projection(self.p3_camera, f=f, c=c)

    @cached_property
    def camera_extrinsics(self):
        from transformations import euler_from_matrix
        camera_params = self.camera_params
        R = np.eye(4)
        R[:3, :3] = camera_params.Rc
        r = np.array(euler_from_matrix(R))
        t = np.squeeze(camera_params.t_c, axis=1)
        return r, t

    @cached_property
    def camera_intrinsics(self):
        camera_params = self.camera_params
        return camera_params.fc, np.squeeze(camera_params.cc, axis=1)


def _apply_to_flattened(data, f):
    shape = data.shape
    data = np.reshape(data, (shape[0]*shape[1],) + shape[2:])
    data = f(data)
    return np.reshape(data, shape)


if __name__ == '__main__':
    subject_code = 'S1'
    sequence_code = 'Walking_1'
    camera_code = 'C1'
    image_frame = 10

    subject = root_node.subject(subject_code)
    sequence = subject.sequence(sequence_code)
    view = sequence.view(camera_code)

    def show():
        from skeleton import s16, s14, native_to_s16, s16_to_s14
        import matplotlib.pyplot as plt

        def convert(native):
            p16 = native_to_s16(native)
            p14 = s16_to_s14(p16)
            return p16, p14

        p3_world_16, p3_world_14 = convert(view.p3_world[image_frame])
        p3_camera_16, p3_camera_14 = convert(view.p3_camera[image_frame])

        p2_16, p2_14 = convert(view.p2[image_frame])

        s16.vis3d(p3_world_16, scatter=False)
        s16.vis3d(p3_camera_16, scatter=False)
        s16.vis(p2_16)
        plt.gca().invert_yaxis()
        plt.show()

        s14.vis3d(p3_world_14, scatter=False)
        s14.vis3d(p3_camera_14, scatter=False)
        s14.vis(p2_14)
        plt.gca().invert_yaxis()
        plt.show()

    def print_views():
        for subject in root_node.subjects:
            for sequence in subject.sequences():
                for view in sequence.views:
                    print(view)

    # print_views()
    show()
