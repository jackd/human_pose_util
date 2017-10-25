from __future__ import division
import numpy as np
from human_pose_util.register import get_skeleton, get_converter
from human_pose_util.transforms import np_impl
from group import copy_group, filter_children, mapped_group, KeySubsetGroup


def normalized_poses(p3, skeleton_id, rotate_front=False, recenter_xy=False):
    """Get a normalized version of p3. Does not change p3."""
    skeleton = get_skeleton(skeleton_id)
    if rotate_front:
        p3 = skeleton.rotate_front(p3)
    if recenter_xy:
        r = skeleton.root_index
        p3 = p3.copy()
        p3[..., :2] -= p3[..., r:r+1, :2]
    return p3


def apply_space_scale(
        sequence, space_scale, div_keys=['p3w', 'p3c'], div_attr_keys=['t']):
    """Apply space rescaling."""
    for k in div_keys:
        if k in sequence:
            sequence[k] = sequence[k] / space_scale
    for k in div_attr_keys:
        if k in sequence.attrs:
            sequence.attrs[k] = sequence.attrs[k] / space_scale
    if 'space_scale' in sequence:
        sequence.attrs['space_scale'] *= space_scale
    else:
        sequence.attrs['space_scale'] = space_scale


def apply_pixel_scale(
        sequence, pixel_scale, div_keys=['p2'], div_attr_keys=['f', 'c']):
    for k in div_keys:
        if k in sequence:
            sequence[k] = sequence[k] / pixel_scale
    for k in div_attr_keys:
        if k in sequence.attrs:
            sequence.attrs[k] = sequence.attrs[k] / pixel_scale
    if 'pixel_scale' in sequence:
        sequence.attrs['pixel_scale'] *= pixel_scale
    else:
        sequence.attrs['pixel_scale'] = pixel_scale


def apply_consistent_pose(sequence):
    p3w = sequence['p3w']
    r, t = (sequence.attrs[k] for k in ['r', 't'])
    sequence['p3c'] = np_impl.transform_frame(p3w, r=r, t=t)


def apply_consistent_projection(sequence):
    p3c = sequence['p3c']
    f, c = (sequence.attrs[k] for k in ['f', 'c'])
    sequence['p2'] = np_impl.project(p3c, f=f, c=c)


def apply_fps_change(sequence, target_fps, seq_keys=['p3c', 'p3w', 'p2']):
    if 'fps' not in sequence.attrs:
        raise KeyError('fps must be in example to change fps.')
    actual_fps = sequence.attrs['fps']
    if actual_fps % target_fps != 0:
        raise ValueError('actual_fps must be divisible by target_fps')
    take_every = actual_fps // target_fps
    for k in seq_keys:
        sequence[k] = sequence[k][::take_every]
    sequence.attrs['fps'] = target_fps


def apply_skeleton_conversion(
        dataset, skeleton_id, pose_keys=['p3c', 'p3w', 'p2']):
    original_skeleton_id = dataset.attrs['skeleton_id']
    if original_skeleton_id == skeleton_id:
        return
    converter = get_converter(original_skeleton_id, skeleton_id)
    for sequence in dataset.values():
        for key in pose_keys:
            if key in sequence:
                sequence[key] = converter.convert(sequence[key])
    dataset.attrs['skeleton_id'] = skeleton_id


def _get_heights(sequences, skeleton):
    p3s = {}
    for sequence in sequences:
        s = sequence.attrs['subject_id']
        if s not in p3s:
            p3s[s] = []
        p3s[s].append(sequence['p3w'])
    heights = {k: np.max(skeleton.height(np.concatenate(v, axis=0)))
               for k, v in p3s.items()}
    return heights


def _str_as_list(x):
    if isinstance(x, (str, unicode)):
        return [x]
    elif hasattr(x, '__iter__'):
        return x
    else:
        raise ValueError('x must be str, unicode or iterable')


def filter_dataset(
        dataset, modes=None, camera_idxs=None, subject_ids=None,
        sequence_ids=None, keys=None):
    """
    Filter a group of sequences.

    Example usage:
        filter_sequences(dataset, subject_ids=['S1'], camera_ids=[1])
        will return sequences with subject_id 'S1' and camera_id 1
    """
    conds = []

    def attr_cond(key, values):
        values = _str_as_list(values)

        def cond(s):
            return s.attrs[key] in values

        return cond

    if subject_ids is not None:
        conds.append(attr_cond('subject_id', subject_ids))
    if sequence_ids is not None:
        conds.append(attr_cond('sequence_id', sequence_ids))
    if camera_idxs is not None:
        camera_idxs = _str_as_list(camera_idxs)
        camera_ids = list(
            set([s.attrs['camera_id'] for k, s in dataset.items()]))
        camera_ids.sort()
        camera_ids = [camera_ids[i] for i in camera_idxs]
        conds.append(attr_cond('camera_id', camera_ids))
    if modes is not None:
        modes = _str_as_list(modes)
        conds.append(lambda x: x.attrs['mode'] in modes)
    if len(conds) > 0:
        dataset = filter_children(
            dataset, lambda k, v: all([cond(v) for cond in conds]))
    if keys is not None:
        dataset = mapped_group(dataset, lambda s: KeySubsetGroup(s, keys))

    return dataset


def normalize_dataset(
        dataset,
        consistent_pose=False, consistent_projection=False,
        scale_to_height=False, space_scale=1, pixel_scale=1, fps=None,
        skeleton_id=None, heights=None):
    """
    Modify data in a dataset.

    If `scale_to_height` is True, applies space_scale after scaling to height,
    i.e. if `space_scale = 5` and the subject height is 1.5, this is equivalent
    to `space_scale = 5*1.5` and no `scale_to_height = False`. Heights will be
    calculated if not provided.

    sequences should be an iterable of pose sequences, each of which is a dict
    with potentially all of the following keys:
        p3w
        r (needed if `consistent_pose`)
        t (needed if `consistent_pose`)
        f (needed if `consistent_projection`)
        c (needed if `consistent_projection`)
        p3c (overwritten if `consistent_pose`)
        p2 (overwritten if `consistent_projection`)
        subject_id (needed if `scale_to_height`)
        fps (needed if `fps is not None`)
        space_scale (optional)
        pixel_scale (optional)
        skeleton_id (needed if `target_skeleton_id is not None`)

    Args:
        ...
        heights: optional dict of subject_id -> heights. Computed if not
            supplied

    Modifies sequences in place.
    """
    if skeleton_id is not None:
        apply_skeleton_conversion(dataset, skeleton_id)

    skeleton_id = dataset.attrs['skeleton_id']
    if scale_to_height and heights is None:
        skeleton = get_skeleton(skeleton_id)
        heights = _get_heights(dataset.values(), skeleton)

    for sequence in dataset.values():
        if consistent_pose:
            apply_consistent_pose(sequence)
        if consistent_projection:
            apply_consistent_projection(sequence)
        if scale_to_height:
            apply_space_scale(
                sequence, heights[sequence.attrs['subject_id']]*space_scale)
        elif space_scale > 1:
            apply_space_scale(sequence, space_scale)

        if pixel_scale != 1:
            apply_pixel_scale(sequence, pixel_scale)

        if fps is not None:
            apply_fps_change(sequence, fps)

    return dataset


# def filter_dataset(dataset, modes=None, camera_idxs=None, keys=None):
#     """
#     Get a view into the dataset containing filtered data.
#
#     Args:
#         modes: iterable of strings
#         camera_idxs: indices of camera_ids (sorted) to be included. Includes
#             all if None
#         keys: keys for which data is returned for each sequence.
#
#     Returns a view into the original dataset exposing only the specified data
#     for the specified sequences.
#     """
#     if modes is not None:
#         if isinstance(modes, (str, unicode)):
#             modes = [modes]
#         dataset = filter_children(
#             dataset, lambda k, v: v.attrs['mode'] in modes)
#     if camera_idxs is not None:
#         camera_ids = list(
#             set([s.attrs['camera_id'] for k, s in dataset.items()]))
#         camera_ids.sort()
#         camera_ids = [camera_ids[i] for i in camera_idxs]
#         dataset = filter_children(
#             dataset, lambda k, v: v.attrs['camera_id'] in camera_ids)
#     if keys is not None:
#         dataset = mapped_group(dataset, lambda s: KeySubsetGroup(s, keys))
#     return dataset


def dataset_to_p3w(
        dataset, rotate_front=False, recenter_xy=False):
    p3 = np.concatenate([s['p3w'] for s in dataset.values()], axis=0)
    if rotate_front or recenter_xy:
        skeleton_id = dataset.attrs['skeleton_id']
        p3 = normalized_poses(p3, skeleton_id, rotate_front, recenter_xy)
    return p3


def dataset_to_view_data(dataset):
    """Get p2, r, t, f, c, p3w from the given dataset."""
    p3w = np.concatenate([s['p3w'] for s in dataset.values()], axis=0)
    p2 = np.concatenate([s['p2'] for s in dataset.values()], axis=0)
    r, t, f, c = [np.concatenate(
        [[s.attrs[k] for i in range(s.attrs['n_frames'])]
         for s in dataset.values()])
        for k in ['r', 't', 'f', 'c']]
    return p2, r, t, f, c, p3w


def normalized_p3w(
        dataset, modes=None, camera_idxs=None, scale_to_height=True,
        fps=None, skeleton_id=None, space_scale=1, rotate_front=True,
        recenter_xy=True):
    """
    Get normalized p3w data from the given dataset.

    Args:
        dataset: base dataset. Will not be changed.
        modes:
        camera_idxs:
        scale_to_height: if True will scale all p3w values to height of subject
        fps: target frames per second.
        skeleton_id: skeleton_id to use. Must be registered.
        space_scale: value to scale spatial values by. If scale_to_height, the
            effects are combined.

    Returns:
        modified_dataset
        p3w, np.ndarray of shape (n_examples, n_joints, 3).
    """
    dataset = copy_group(filter_dataset(
        dataset, modes=modes, camera_idxs=camera_idxs, keys=['p3w']))
    normalize_dataset(
            dataset, scale_to_height=scale_to_height, space_scale=space_scale,
            skeleton_id=skeleton_id)
    skeleton_id = dataset.attrs['skeleton_id']
    return dataset, dataset_to_p3w(
        dataset, rotate_front=rotate_front, recenter_xy=recenter_xy)


def normalized_view_data(
        dataset, modes=None, camera_idxs=None, keys=None,
        skeleton_id=None,
        pixel_scale=1000, space_scale=1000, consistent_pose=True,
        consistent_projection=True, fps=None):
    """
    Get normalized data relevant to view/camera.

    Args:
        dataset: filtered dataset. Will not be changed.
        skeleton_id:
        pixel_scale: value to scale pixel values by
        space_scale: value to scale

    Returns:
        modified_dataset_attrs
        tuple: p2, r, t, f, c, p3w
    """
    keys = ['p3w']
    if not consistent_pose:
        keys.append('p3c')
    if not consistent_projection:
        keys.append('p2')
    dataset = copy_group(filter_dataset(
        dataset, modes=modes, camera_idxs=camera_idxs, keys=['p3w']))
    normalize_dataset(
        dataset, space_scale=space_scale, pixel_scale=pixel_scale,
        consistent_pose=consistent_pose,
        consistent_projection=consistent_projection,
        fps=fps, skeleton_id=skeleton_id)
    return dataset, dataset_to_view_data(dataset)
