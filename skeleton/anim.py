from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from . import vis


class SkeletonAnimator(object):
    def __init__(self, skeleton, poses, kwargs_fn=None, **constant_kwargs):
        self.check_pose_shape(poses.shape)
        self._poses = poses
        self._skeleton = skeleton
        self._kwargs_fn = kwargs_fn
        self._constant_kwargs = constant_kwargs

    def check_pose_shape(self, shape):
        pass

    def init(self, skeleton, ax, pose0):
        raise NotImplementedError()

    def animate(self, skeleton, pose):
        raise NotImplementedError()

    def set_lims(self, ax, mins, maxs):
        raise NotImplementedError()

    @property
    def skeleton(self):
        return self._skeleton

    def get_animation(self, fps=50, fig=None):
        interval = 1000 // fps
        skeleton = self.skeleton

        def init_fn():
            print('init')
            if fig is None:
                f = plt.figure()
                ax = f.gca()
                mins = np.min(self._poses, axis=(0, 1))
                maxs = np.max(self._poses, axis=(0, 1))
                max_range = np.max(maxs - mins)
                centers = (mins + maxs) / 2
                r = max_range / 2
                mins = centers - r
                maxs = centers + r
                self.set_lims(ax, mins, maxs)
                # ax.set_aspect('equal', 'datalim')
                ax.invert_yaxis()
            else:
                f = fig
                ax = f.gca()
            self._fig = f
            self.init(skeleton, ax, self._poses[0])
            return tuple(self._lines)

        def animate_fn(i):
            print('ani')
            self.animate(skeleton, self._poses[i])
            return tuple(self._lines)

        # init_fn()

        return anim.FuncAnimation(
            fig, animate_fn, range(len(self._poses)), interval=interval,
            init_func=init_fn
        )


class SkeletonAnimator2D(SkeletonAnimator):
    def check_pose_shape(self, shape):
        if len(shape) != 3:
            raise ValueError('poses must be rank 3')
        if shape[2] != 2:
            raise ValueError('poses must be 2D')

    def init(self, skeleton, ax, pose0):
        self._lines = vis.get_lines2d(ax, self.skeleton, pose0)

    def animate(self, skeleton, pose):
        vis.update_lines2d(skeleton, self._lines, pose)

    def set_lims(self, ax, mins, maxs):
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])


class SkeletonAnimator3D(SkeletonAnimator):
    def check_pose_shape(self, shape):
        if len(shape) != 3:
            raise ValueError('poses must be rank 3')
        if shape[2] != 3:
            raise ValueError('poses must be 3D')

    def init(self, skeleton, ax, pose0):
        self._lines = vis.get_lines3d(ax, self.skeleton, pose0)

    def animate(self, skeleton, pose):
        vis.update_lines3d(skeleton, self._lines, pose)

    def set_lims(self, ax, mins, maxs):
        print(mins)
        print(maxs)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])


def get_animation(
        skeleton, points, fig=None, kwargs_fn=None, fps=50, **constant_kwargs):
    dims = points.shape[-1]
    if dims == 2:
        constructor = SkeletonAnimator2D
    elif dims == 3:
        constructor = SkeletonAnimator3D
    else:
        raise ValueError(
            'points must be 3D, got shape "%s"' % str(points.shape))
    return constructor(
        skeleton, points, kwargs_fn=kwargs_fn,
        **constant_kwargs).get_animation(fps=fps, fig=fig)
