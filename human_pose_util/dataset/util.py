"""."""
import numpy as np
from human_pose_util.register import skeleton_register
from human_pose_util.skeleton import front_angle
from human_pose_util.transforms.np_impl import rotate_about


def sequences_to_poses(dataset):
    p3s = []
    for key in dataset:
        p3 = np.array(dataset[key]['p3w'], dtype=np.float32)
        p3s.append(p3)
    return np.concatenate(p3s, axis=0)


def sequences_to_rescaled_poses(dataset):
    from human_pose_util.dataset.spec import calculate_heights
    heights = calculate_heights(dataset)
    p3s = []
    for key in dataset:
        example = dataset[key]
        height = heights[example.attrs['subject_id']]
        p3 = np.array(example['p3w'], dtype=np.float32)
        p3 /= height
        p3s.append(p3)
    p3s = np.concatenate(p3s, axis=0)
    return p3s


def normalized_poses(dataset, scale=True, center=True, rotate_front=True):
    if scale:
        p3s = sequences_to_rescaled_poses(dataset)
    else:
        p3s = sequences_to_poses(dataset)
    skeleton = skeleton_register[dataset.attrs['skeleton_id']]

    if center:
        # hips above origin
        r = skeleton.joint_index(skeleton.root_joint)
        p3s[..., :2] -= p3s[:, r:r+1, :2]
    if rotate_front:
        # rotate hips to front
        phi = front_angle(p3s, skeleton)
        p3s = rotate_about(p3s, -np.expand_dims(phi, axis=1), dim=2)
    return p3s
