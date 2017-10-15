import numpy as np
from human_pose_util.skeleton import \
    Skeleton, SkeletonConverter, skeleton_height, front_angle

head = 'head'
neck = 'neck'
thorax = 'thorax'
pelvis = 'pelvis'
r_shoulder = 'r_shoulder'
r_elbow = 'r_elbow'
r_wrist = 'r_wrist'
l_shoulder = 'l_shoulder'
l_elbow = 'l_elbow'
l_wrist = 'l_wrist'
r_hip = 'r_hip'
r_knee = 'r_knee'
r_ankle = 'r_ankle'
l_hip = 'l_hip'
l_knee = 'l_knee'
l_ankle = 'l_ankle'


class _S14(Skeleton):
    def __init__(self):
        links = (
            (head, None),
            (neck, head),
            (r_shoulder, neck),
            (r_elbow, r_shoulder),
            (r_wrist, r_elbow),
            (l_shoulder, neck),
            (l_elbow, l_shoulder),
            (l_wrist, l_elbow),
            (r_hip, neck),
            (r_knee, r_hip),
            (r_ankle, r_knee),
            (l_hip, neck),
            (l_knee, l_hip),
            (l_ankle, l_knee),
        )
        assert(len(links) == 14)
        super(_S14, self).__init__(links)

    def height(self, p3):
        return skeleton_height(
            p3, l_foot_index=self.joint_index(l_ankle),
            r_foot_index=self.joint_index(r_ankle),
            head_index=self.joint_index(head))

    def front_angle(self, p3):
        return front_angle(self, p3, l_joint=l_hip, r_joint=r_hip)


s14 = _S14()


class _S16(Skeleton):
    def __init__(self):
        links = (
            (head, None),
            (neck, head),
            (thorax, neck),
            (pelvis, thorax),
            (r_shoulder, neck),
            (r_elbow, r_shoulder),
            (r_wrist, r_elbow),
            (l_shoulder, neck),
            (l_elbow, l_shoulder),
            (l_wrist, l_elbow),
            (r_hip, pelvis),
            (r_knee, r_hip),
            (r_ankle, r_knee),
            (l_hip, pelvis),
            (l_knee, l_hip),
            (l_ankle, l_knee),
        )
        assert(len(links) == 16)
        super(_S16, self).__init__(links)

    def height(self, p3):
        return skeleton_height(
            p3, l_foot_index=self.joint_index(l_ankle),
            r_foot_index=self.joint_index(r_ankle),
            head_index=self.joint_index(head))

    def front_angle(self, p3):
        return front_angle(self, p3, l_joint=l_hip, r_joint=r_hip)

    # def to_s14(self, data):
    #     assert(data.shape[-2] == self.n_joints)
    #     if not hasattr(self, '_to_s14_indices'):
    #         _s16_to_s14_indices = []
    #         for joint in s14.joints:
    #             _s16_to_s14_indices.append(self.joint_index(joint))
    #         self._to_s14_indices = np.array(_s16_to_s14_indices)
    #
    #     return data[..., self._to_s14_indices, :]


s16 = _S16()


head_distal = 'head_distal'
head_proximal = 'head_proximal'
l_lower_arm_distal = 'l_lower_arm_distal'
l_lower_arm_proximal = 'l_lower_arm_proximal'
l_lower_leg_distal = 'l_lower_leg_distal'
l_lower_leg_proximal = 'l_lower_leg_proximal'
r_lower_arm_distal = 'r_lower_arm_distal'
r_lower_arm_proximal = 'r_lower_arm_proximal'
r_lower_leg_distal = 'r_lower_leg_distal'
r_lower_leg_proximal = 'r_lower_leg_proximal'
torso_distal = 'torso_distal'
torso_proximal = 'torso_proximal'
l_upper_arm_distal = 'l_upper_arm_distal'
l_upper_arm_proximal = 'l_upper_arm_proximal'
l_upper_leg_distal = 'l_upper_leg_distal'
l_upper_leg_proximal = 'l_upper_leg_proximal'
r_upper_arm_distal = 'r_upper_arm_distal'
r_upper_arm_proximal = 'r_upper_arm_proximal'
r_upper_leg_distal = 'r_upper_leg_distal'
r_upper_leg_proximal = 'r_upper_leg_proximal'


class _S20(Skeleton):
    def __init__(self):
        links = (
            (head_distal, None),
            (head_proximal, None),
            (l_lower_arm_distal, None),
            (l_lower_arm_proximal, None),
            (l_lower_leg_distal, None),
            (l_lower_leg_proximal, None),
            (r_lower_arm_distal, None),
            (r_lower_arm_proximal, None),
            (r_lower_leg_distal, None),
            (r_lower_leg_proximal, None),
            (torso_distal, None),
            (torso_proximal, None),
            (l_upper_arm_distal, None),
            (l_upper_arm_proximal, None),
            (l_upper_leg_distal, None),
            (l_upper_leg_proximal, None),
            (r_upper_arm_distal, None),
            (r_upper_arm_proximal, None),
            (r_upper_leg_distal, None),
            (r_upper_leg_proximal, None),
        )
        assert(len(links) == 20)
        super(_S20, self).__init__(links)

    def _calc_raw_indices(self):
        """Used in `from_raw`."""
        _native_map = {
            head_distal: 'headDistal',
            head_proximal: 'headProximal',
            l_lower_arm_distal: 'lowerLArmDistal',
            l_lower_arm_proximal: 'lowerLArmProximal',
            l_lower_leg_distal: 'lowerLLegDistal',
            l_lower_leg_proximal: 'lowerLLegProximal',
            r_lower_arm_distal: 'lowerRArmDistal',
            r_lower_arm_proximal: 'lowerRArmProximal',
            r_lower_leg_distal: 'lowerRLegDistal',
            r_lower_leg_proximal: 'lowerRLegProximal',
            torso_distal: 'torsoDistal',
            torso_proximal: 'torsoProximal',
            l_upper_arm_distal: 'upperLArmDistal',
            l_upper_arm_proximal: 'upperLArmProximal',
            l_upper_leg_distal: 'upperLLegDistal',
            l_upper_leg_proximal: 'upperLLegProximal',
            r_upper_arm_distal: 'upperRArmDistal',
            r_upper_arm_proximal: 'upperRArmProximal',
            r_upper_leg_distal: 'upperRLegDistal',
            r_upper_leg_proximal: 'upperRLegProximal',
        }
        self._raw_indices = [_native_map[k] for k in self.joints]

    def from_raw(self, pose):
        """Convert raw data to (Pose) to s20 data."""
        if not hasattr(self, '_raw_indices'):
            self._calc_raw_indices()
        return np.stack(
            [np.array([i[0][0, 0] for i in pose[k]])
             for k in self._raw_indices], axis=0)


s20 = _S20()
_s16_to_s14_converter = SkeletonConverter(s16, s14)


def s16_to_s14_converter():
    return _s16_to_s14_converter


class _S20ToS16Converter(SkeletonConverter):
    def __init__(self):
        self._calc_mid_indices()
        self._calc_common_indices()

    def _calc_mid_indices(self):
        """Used in `to_s16`."""
        midpoints = (
            (l_elbow, [l_upper_arm_distal, l_lower_arm_proximal]),
            (r_elbow, [r_upper_arm_distal, r_lower_arm_proximal]),
            (l_knee, [l_upper_leg_distal, l_lower_leg_proximal]),
            (r_knee, [r_upper_leg_distal, r_lower_leg_proximal]),
        )
        s16_indices = []
        n_indices = []
        for s16_joint, ns in midpoints:
            s16_indices.append(s16.joint_index(s16_joint))
            n_indices.append([s20.joint_index(n) for n in ns])
        self._s16_s20_mid = np.array(s16_indices, dtype=np.int32)
        self._s20_s16_mid = np.array(n_indices, dtype=np.int32)

    def _calc_common_indices(self):
        """Used in `to_s16`."""
        common = (
            (head, head_distal),
            (neck, head_proximal),
            (thorax, torso_proximal),
            (pelvis, torso_distal),
            (l_shoulder, l_upper_arm_proximal),
            (l_wrist,  l_lower_arm_distal),
            (r_shoulder, r_upper_arm_proximal),
            (r_wrist, r_lower_arm_distal),
            (l_hip, l_upper_leg_proximal),
            (l_ankle, l_lower_leg_distal),
            (r_hip, r_upper_leg_proximal),
            (r_ankle, r_lower_leg_distal)
         )
        s16_indices = []
        n_indices = []
        for s16_joint, n in common:
            s16_indices.append(s16.joint_index(s16_joint))
            n_indices.append(s20.joint_index(n))
        self._s16_s20_common = np.array(s16_indices, dtype=np.int32)
        self._s20_s16_common = np.array(n_indices, dtype=np.int32)

    def convert(self, data):
        shape = list(data.shape)
        if shape[-2] != 20:
            raise Exception(
                'Invalid number of joints for native: %d' % shape[-2])
        shape[-2] = 16
        s16_data = np.empty(shape, dtype=data.dtype)
        s16_data[..., self._s16_s20_common, :] = \
            data[..., self._s20_s16_common, :]
        s16_data[..., self._s16_s20_mid, :] = (
            data[..., self._s20_s16_mid[:, 0], :] +
            data[..., self._s20_s16_mid[:, 1], :]) / 2
        return s16_data

    def convert_tf(self, data):
        """
        Not implemented.

        Should be able to do something similar to `convert` with
        `tf.dynamic_stitch`, but this isn't a priority right now.
        """
        raise NotImplementedError()


_s20_to_s16_converter = _S20ToS16Converter()


def s20_to_s16_converter():
    return _s20_to_s16_converter


class _S20ToS14Converter(SkeletonConverter):
    def convert(self, data):
        return _s16_to_s14_converter.convert(
            _s20_to_s16_converter.convert(data))

    def __init__(self):
        pass


_s20_to_s14_converter = _S20ToS14Converter()


def s20_to_s14_converter():
    return _s20_to_s14_converter
