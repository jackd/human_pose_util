import numpy as np
from stateful import Stateful, StatefulAnimator
from glumpy.graphics.collections import SegmentCollection


_side_index = {
    'l': 0,
    'r': 1,
    'c': 2
}
_n_sides = len(_side_index)


def joint_side(joint):
    if joint[:2] == 'l_':
        return 'l'
    elif joint[:2] == 'r_':
        return 'r'
    else:
        return 'c'


class LimbCollection(Stateful):
    """Stateful objected with pose as state."""
    def __init__(self, skeleton, pose, transform, viewport, linewidth=2.0):
        children = [[] for k in _side_index]
        parents = [[] for k in _side_index]

        for c in range(skeleton.n_joints):
            p = skeleton.parent_index(c)
            if p is not None:
                side = joint_side(skeleton.joint(c))
                index = _side_index[side]
                children[index].append(c)
                parents[index].append(p)
        self.children = children
        self.parents = parents

        self.body_segments = [
            SegmentCollection(
                mode="agg", transform=transform, viewport=viewport,
                linewidth='local', color='local') for _ in range(3)]

        for i, segment in enumerate(self.body_segments):
            segment.append(
                pose[children[i]], pose[parents[i]],
                linewidth=linewidth)

        self.body_segments[_side_index['l']]['color'] = 0, 0, 1, 1
        self.body_segments[_side_index['r']]['color'] = 1, 0, 0, 1
        # self.body_segments[side_index['c']]['color'] = 0, 0, 0, 1
        super(LimbCollection, self).__init__(pose)

    def draw(self):
        for segments in self.body_segments:
            segments.draw()

    @property
    def pose(self):
        return self.state

    @pose.setter
    def pose(self, new_pose):
        self.state = new_pose

    def update(self, old_state, new_state):
        for i, segment in enumerate(self.body_segments):
            segment['P0'] = np.repeat(
                new_state[self.children[i]], 4, axis=0)
            segment['P1'] = np.repeat(
                new_state[self.parents[i]], 4, axis=0)


def limb_collection_animator(limb_collection, poses, fps):
    n_frames = len(poses) - 1

    def state_fn(time):
        f = time * fps % n_frames
        f0 = int(f)
        frac = f % 1
        return (1 - frac) * poses[f0] + frac*poses[f0+1]

    return StatefulAnimator(limb_collection, state_fn)


def skeleton_animator(
        skeleton, poses, fps, transform, viewport, linewidth=2.0):
    limb_collection = LimbCollection(
        skeleton, poses[0], transform, viewport, linewidth=linewidth)
    return limb_collection_animator(limb_collection, poses, fps)
