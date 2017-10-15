"""Provides skeleton class for visualizations."""
import numpy as np
import matplotlib.pyplot as plt


def default_ax2d(fig=None):
    """A good default axis to use for single plots of 2d data."""
    fig = fig or plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal', 'datalim')
    return ax


def default_ax3d(fig=None):
    """A good default 3D axis for 3D skeleton data."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = fig or plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal', 'datalim')
    # ax.view_init(0, 0)
    return ax


def rescale_ax3d(ax, x, y, z):
    """Rescale a 3d axis to given points."""
    def _center_width(x):
        mi = np.min(x)
        ma = np.max(x)
        center = (mi + ma) / 2
        width = ma - mi
        return center, width

    xc, xw = _center_width(x)
    yc, yw = _center_width(y)
    zc, zw = _center_width(z)

    r = max(xw, yw, zw) / 2
    ax.set_xlim(xc - r, xc + r)
    ax.set_ylim(yc - r, yc + r)
    ax.set_zlim(zc - r, zc + r)


def _color(joint_id):
    if joint_id[0] == 'l':
        return 'blue'
    elif joint_id[0] == 'r':
        return 'red'
    else:
        return 'black'


def _plot_kwargs(child_id, constant_kwargs, kwargs_fn=None):
    kwargs = constant_kwargs.copy()
    if kwargs_fn is not None:
        kwargs = kwargs_fn(child_id)
    if 'color' not in kwargs:
        kwargs['color'] = _color(child_id)
    return kwargs


def vis2d(skeleton, points, ax=None, change_ax=True, kwargs_fn=None,
          **constant_kwargs):
    """Visualize 2D skeleton data using matplotlib.pyplot."""
    if points.shape != (skeleton.n_joints, 2):
        raise ValueError('Expected points of shape %s, got %s'
                         % ((skeleton.n_joints, 2), points.shape))
    ax = ax or default_ax2d()
    if change_ax:
        # ax.invert_yaxis()
        ax.set_aspect('equal', 'datalim')
    for child in range(skeleton.n_joints):
        parent = skeleton.parent_index(child)
        if parent is not None:
            xy = points[[child, parent]]
            kwargs = _plot_kwargs(
                skeleton.joint(child), constant_kwargs, kwargs_fn)
            ax.plot(xy[:, 0], xy[:, 1], **kwargs)
    return ax


def vis3d(skeleton, points, ax=None, change_ax=True, kwargs_fn=None,
          **constant_kwargs):
    """Visualize 3D skeleton data using matplotlib.pyplot."""
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    if points.shape != (skeleton.n_joints, 3):
        raise ValueError('Expected points of shape %s, got %s'
                         % ((skeleton.n_joints, 3), points.shape))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    if ax is None:
        ax = default_ax3d()

    if change_ax:
        rescale_ax3d(ax, x, y, z)

    for child in range(skeleton.n_joints):
        parent = skeleton.parent_index(child)
        if parent is not None:
            kwargs = _plot_kwargs(
                skeleton.joint(child), constant_kwargs, kwargs_fn)
            ax.plot(
                x[[child, parent]], y[[child, parent]], zs=z[[child, parent]],
                **kwargs)
    return ax
