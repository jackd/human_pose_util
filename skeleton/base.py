import numpy as np


class Skeleton(object):
    def __init__(self, child_parent_links):
        """
        Create the skeleton with the given id and child parent links.

        child_parent_links should be a list/tuple of tuples, where each element
        is (child, parent) of a link. Root joint(s) may have None parents.
        """
        self._child_parent_links = child_parent_links
        self._indices = {}
        self._root_joints = []
        for i, (child, parent) in enumerate(child_parent_links):
            if parent is None:
                self._root_joints.append(child)
            self._indices[child] = i
        if len(self._root_joints) == 0:
            raise Exception('Skeleton must have at least 1 root joint')
        self._parent_indices = [
            self._indices[parent] if parent is not None else None
            for i, (c, parent) in enumerate(child_parent_links)]
        self._n_joints = len(child_parent_links)

    @property
    def root_joint(self):
        if len(self._root_joints) > 1:
            raise Exception('Skeleton has multiple root joints')
        else:
            assert(len(self._root_joints) == 1)
            return self._root_joints[0]

    @property
    def skeleton_id(self):
        return self._skeleton_id

    @property
    def n_joints(self):
        return self._n_joints

    @property
    def joints(self):
        return [c[0] for c in self._child_parent_links]

    def joint_index(self, joint_id):
        return self._indices[joint_id]

    def has_joint(self, joint):
        return joint in self._indices

    def joint(self, joint_index):
        return self._child_parent_links[joint_index][0]

    def parent(self, joint_id):
        return self._joint_id(self.parent_index(self.joint_index(joint_id)))

    def parent_index(self, joint_index):
        return self._parent_indices[joint_index]

    @property
    def link_indices(self):
        """Get (children, parents) for all links with non-None parents."""
        return zip(
            *[(c, p) for c, p in self._child_parent_links if p is not None])

    def symmetric_id(self, joint_id):
        """
        Get the id of the symmetric joint.

        The symmetric joint of a joint starting with 'l' is the corresponding
        version starting with 'r', i.e. 'r_ankle' -> 'l_ankle' and vice versa.
        """
        if joint_id[:2] == 'l_':
            return 'r%s' % joint_id[1:]
        elif joint_id[:2] == 'r_':
            return 'l%s' % joint_id[1:]
        else:
            return joint_id

    def symmetric_index(self, joint_index):
        """Index version of 'symmetric_id'."""
        return self.joint_index(self.symmetric_id(self.joint_id(joint_index)))

    def symmetric_indices(self):
        """Get all symmetric indices for this skeleton."""
        return [self.symmetric_index(j) for j in range(self.n_joints)]

    def reflected_points(self, points):
        """
        Get the symmetric version of all points.

        Just reorders the 2nd last axis - does not change values.
        """
        assert(points.shape[-2] == self._n_joints)
        return points[..., self.symmetric_indices(), :]

    def height(self, p3):
        raise NotImplementedError()

    def front_angle(self, p3):
        raise NotImplementedError()


class SkeletonConverter(object):
    def __init__(self, input_skeleton, output_skeleton, oi_map={}):
        indices = []
        for i in range(output_skeleton.n_joints):
            joint = output_skeleton.joint(i)
            if joint in oi_map:
                joint = oi_map[joint]
            indices.append(input_skeleton.joint_index(joint))
        self._indices = indices

    def convert(self, input_data):
        return input_data[..., self._indices, :]

    def convert_tf(self, input_data):
        import tensorflow as tf
        return tf.gather(input_data, self._indices, axis=-2)


def skeleton_height(p3, l_foot_index, r_foot_index, head_index):
    displ = (p3[:, l_foot_index] + p3[:, r_foot_index]) / 2 - p3[:, head_index]
    return np.sqrt(np.max(np.sum(displ**2, axis=-1)))


def front_angle(points, skeleton, l_joint='l_hip', r_joint='r_hip'):
    """Get the angle about the z-axis of the hips."""
    import numpy as np
    li = skeleton.joint_index(l_joint)
    ri = skeleton.joint_index(r_joint)
    diff = points[..., li, :] - points[..., ri, :]
    return np.arctan2(diff[..., 1], diff[..., 0])
