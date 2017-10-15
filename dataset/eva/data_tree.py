"""Provides classes for accessing data like a tree."""
from human_eva.path import video_path
from human_eva.meta import partitions, n_frames


class TreeNode(object):
    @property
    def parent(self):
        raise NotImplementedError()

    @property
    def children(self):
        raise NotImplementedError()

    @property
    def code(self):
        raise NotImplementedError()

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root

    def __str__(self):
        return '%s/%s' % (self.parent, self.code)

    def __repr__(self):
        return '%s/%s' % (self.parent, self.code)


class RootNode(TreeNode):
    def subject(self, subject_code):
        raise NotImplementedError()

    @property
    def subject_codes(self):
        raise NotImplementedError()

    @property
    def subjects(self):
        return [self.subject(s) for s in self.subject_codes]

    @property
    def children(self):
        return self.subjects

    @property
    def parent(self):
        return None

    def sequence(self, subject_code, sequence_code):
        return self.subject(subject_code).sequence(sequence_code)

    def view(self, subject_code, sequence_code, camera_code):
        return self.subject(subject_code).view(sequence_code, camera_code)

    def __str__(self):
        return 'DataTree: '

    def __repr__(self):
        return 'DataTree: '


class SubjectNode(TreeNode):
    def sequence(self, sequence_code):
        raise NotImplementedError()

    @property
    def parent(self):
        """
        Get the root node for this implementation.

        Should satisfy self.parent.subject(self.subject_code) == self.
        """
        raise NotImplementedError()

    @property
    def subject_code(self):
        raise NotImplementedError()

    def sequence_codes(self, base_code=None):
        raise NotImplementedError()

    def view(self, sequence_code, camera_code):
        return self.sequence(sequence_code).view(camera_code)

    @property
    def code(self):
        return self.subject_code

    @property
    def children(self):
        return self.sequences()

    def sequences(self, base_code=None):
        return [self.sequence(c) for c in self.sequence_codes(
            base_code=base_code)]


class SequenceNode(TreeNode):

    @property
    def subject(self):
        raise NotImplementedError()

    def view(self, camera_code):
        raise NotImplementedError()

    @property
    def camera_codes(self):
        raise NotImplementedError()

    @property
    def sequence_code(self):
        raise NotImplementedError()

    @property
    def parent(self):
        return self.subject

    @property
    def children(self):
        return self.views

    @property
    def views(self):
        return [self.view(c) for c in self.camera_codes]

    @property
    def subject_code(self):
        return self.subject.code

    @property
    def code(self):
        return self.sequence_code

    @property
    def base_sequence_code(self):
        return self.sequence_code.split('_')[0]

    @property
    def trial_index(self):
        return int(self.sequence_code.split('_')[1]) - 1

    @property
    def has_joint_data(self):
        return self.trial_index != 1 and self.base_sequence_code != 'Combo' \
            and self.subject_code != 'S4'

    @property
    def has_video(self):
        return self.trial_index != 2

    @property
    def fps(self):
        return 120 if self.trial_index == 2 else 60

    @property
    def n_frames(self):
        return n_frames[
            self.subject_code][self.base_sequence_code][self.trial_index]

    @property
    def train_frame_partition(self):
        if self.trial_index == 0:
            return partitions[self.subject_code][self.base_sequence_code]
        elif self.trial_index == 2:
            return (0, self.n_frames)
        else:
            raise Exception('No train data for sequence %s' % self.code)


class ViewNode(TreeNode):

    @property
    def sequence(self):
        raise NotImplementedError()

    @property
    def camera_code(self):
        raise NotImplementedError()

    @property
    def p3_camera(self):
        """
        The 3D coordinates of each joint in camera coordinates.

        Equivalent to the transformation of p3_world according to
        camera_extrinsics.
        """
        raise NotImplementedError()

    @property
    def p3_world(self):
        """
        The 3D coordinates of each joint in world coordinates.

        Equivalent to the inverse transformation of p3_camera according to
        camera_extrinsics.
        """
        raise NotImplementedError()

    @property
    def p2(self):
        """
        The 2D image coordinates of each joint.

        Equivalent to the projection of p3 using camera_intrinsics.
        """
        raise NotImplementedError()

    @property
    def camera_extrinsics(self):
        """Get the camera extrinsic parameters: (r, t)."""
        raise NotImplementedError()

    @property
    def camera_intrinsics(self):
        """Get the camera intrinsic parameters: (f, c)."""
        raise NotImplementedError()

    @property
    def n_frames(self):
        return self.sequence.n_frames

    @property
    def parent(self):
        return self.sequence

    @property
    def subject(self):
        return self.sequence.subject

    @property
    def subject_code(self):
        return self.subject.code

    @property
    def sequence_code(self):
        return self.sequence.code

    @property
    def code(self):
        return self.camera_code

    @property
    def video_path(self):
        return video_path(
            self.subject_code, self.sequence_code, self.camera_code)
