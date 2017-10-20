from __future__ import division
import numpy as np
from human_pose_util.serialization import skeleton_register, converter_register
from human_pose_util.transforms import np_impl


def normalized_poses(p3, skeleton_id, rotate_front=False, recenter_xy=False):
    """Get a normalized version of p3. Does not change p3."""
    skeleton = skeleton_register[skeleton_id]
    if rotate_front:
        p3 = skeleton.rotate_front(p3)
    if recenter_xy:
        r = skeleton.root_index
        p3 = p3.copy()
        p3[..., :2] -= p3[..., r:r+1, :2]
    return p3


def apply_space_scale(
        example, space_scale, div_keys=['p3w', 'p3c', 't']):
    """Apply space rescaling."""
    for k in div_keys:
        if k in example:
            example[k] = example[k] / space_scale
    if 'space_scale' in example:
        example['space_scale'] *= space_scale
    else:
        example['space_scale'] = space_scale


def apply_pixel_scale(example, pixel_scale, div_keys=['p2', 'f', 'c']):
    for k in div_keys:
        if k in example:
            example[k] = example[k] / pixel_scale
    if 'pixel_scale' in example:
        example['pixel_scale'] *= pixel_scale
    else:
        example['pixel_scale'] = pixel_scale


def apply_consistent_pose(example):
    p3w, r, t = (example[k] for k in ['p3w', 'r', 't'])
    example['p3c'] = np_impl.transform_frame(p3w, r=r, t=t)


def apply_consistent_projection(example):
    p3c, f, c = (example[k] for k in ['p3c', 'f', 'c'])
    example['p2'] = np_impl.project(p3c, f=f, c=c)


def apply_fps_change(example, target_fps, seq_keys=['p3c', 'p3w', 'p2']):
    if 'fps' not in example:
        raise KeyError('fps must be in example to change fps.')
    actual_fps = example['fps']
    if actual_fps % target_fps != 0:
        raise ValueError('actual_fps must be divisible by target_fps')
    take_every = actual_fps // target_fps
    for k in seq_keys:
        if k in example:
            example[k] = example[k][::take_every]


def apply_skeleton_conversion(
        example, target_skeleton_id, pose_keys=['p3c', 'p3w', 'p2']):
    skeleton_id = example['skeleton_id']
    if skeleton_id == target_skeleton_id:
        return
    converter = converter_register[(skeleton_id, target_skeleton_id)]
    for key in pose_keys:
        if key in example:
            example[key] = converter.convert(example[key])
    example['skeleton_id'] = target_skeleton_id


def _get_heights(examples, skeleton):
    p3s = {}
    for example in examples:
        s = example['subject_id']
        if s not in p3s:
            p3s[s] = []
        p3s[s].append(example['p3w'])
    heights = {k: np.max(skeleton.height(np.concatenate(v, axis=0)))
               for k, v in p3s.items()}
    return heights


def filter_sequences(
        sequences, subject_ids=None, sequence_ids=None, camera_ids=None):
    """
    Filter an iterable of sequences.

    Example usage:
        filter_sequences(sequences, subject_ids=['S1'], camera_ids=[1])
        will return sequences with subject_id 'S1' and camera_id 1
    """
    conds = []
    if subject_ids is not None:
        conds.append(lambda x: x['subject_id'] in subject_ids)
    if sequence_ids is not None:
        conds.append(lambda x: x['sequence_id'] in sequence_ids)
    if camera_ids is not None:
        conds.append(lambda x: x['camera_id'] in camera_ids)
    return [s for s in sequences if all([c(s) for c in conds])]


def normalize_sequences(
        sequences,
        consistent_pose=False, consistent_projection=False,
        scale_to_height=False, space_scale=1, pixel_scale=1, fps=None,
        target_skeleton_id=None, heights=None):
    """
    Modify data in sequences.

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

    Modifies sequences in place.
    """
    if scale_to_height and heights is None:
        skeleton_id = sequences[0]['skeleton_id']
        if not all([seq['skeleton_id'] == skeleton_id for seq in sequences]):
            raise NotImplementedError()
        skeleton = skeleton_register[skeleton_id]
        heights = _get_heights(sequences, skeleton)

    for sequence in sequences:
        if consistent_pose:
            apply_consistent_pose(sequence)
        if consistent_projection:
            apply_consistent_projection(sequence)
        if scale_to_height:
            apply_space_scale(
                sequence, heights[sequence['subject_id']]*space_scale)
        elif space_scale > 1:
            apply_space_scale(sequence, space_scale)

        if pixel_scale != 1:
            apply_pixel_scale(sequence, pixel_scale)

        if fps is not None:
            apply_fps_change(sequence, fps)

        if target_skeleton_id is not None:
            apply_skeleton_conversion(sequence, target_skeleton_id)

    return sequences
