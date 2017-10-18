from human_pose_util.skeleton import Skeleton, SkeletonConverter

spine3 = 'spine3'
spine4 = 'spine4'
spine2 = 'spine2'
spine = 'spine'
pelvis = 'pelvis'
neck = 'neck'
head = 'head'
head_top = 'head_top'
l_clavicle = 'l_clavicle'
l_shoulder = 'l_shoulder'
l_elbow = 'l_elbow'
l_wrist = 'l_wrist'
l_hand = 'l_hand'
r_clavicle = 'r_clavicle'
r_shoulder = 'r_shoulder'
r_elbow = 'r_elbow'
r_wrist = 'r_wrist'
r_hand = 'r_hand'
l_hip = 'l_hip'
l_knee = 'l_knee'
l_ankle = 'l_ankle'
l_foot = 'l_foot'
l_toe = 'l_toe'
r_hip = 'r_hip'
r_knee = 'r_knee'
r_ankle = 'r_ankle'
r_foot = 'r_foot'
r_toe = 'r_toe'

all_joint_names = (
    spine3,
    spine4,
    spine2,
    spine,
    pelvis,
    neck,
    head,
    head_top,
    l_clavicle,
    l_shoulder,
    l_elbow,
    l_wrist,
    l_hand,
    r_clavicle,
    r_shoulder,
    r_elbow,
    r_wrist,
    r_hand,
    l_hip,
    l_knee,
    l_ankle,
    l_foot,
    l_toe,
    r_hip,
    r_knee,
    r_ankle,
    r_foot,
    r_toe,
)

n_base_joints = len(all_joint_names)
assert(n_base_joints == 28)


def _zero_based(vals):
    """Make a 1-based iterable into a 0-based list."""
    return [v - 1 for v in vals]


def _transform(matlab_dict):
    """Transform a 1-based list into a zero-based tuple."""
    return {k: tuple(_zero_based(v)) for k, v in matlab_dict.items()}


_joint_idx = _transform({
    'all': range(1, 29),
    'relevant':
        [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7],
    'extended':
        [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7, 18,
         13, 28, 23]
})

_parents_o1 = _transform({
    'all':
        [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19,
         20, 21, 22, 5, 24, 25, 26, 27],
    'relevant':
        [2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2],
    'extended':
        [2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2, 5, 8, 11,
         14]
})

_parents_o2 = _transform({
    'all': [4, 3, 5, 5, 5, 1, 2, 6, 2, 6, 9, 10, 11, 2, 6, 14, 15, 16, 4, 5,
            19, 20, 21, 4, 5, 24, 25, 26],
    'relevant':
        [16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16],
    'extended':
        [16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16, 4, 7,
         10, 13]
})

_parents = {'o1': _parents_o1, 'o2': _parents_o2}


def skeleton(joint_set_name='relevant', order='o1'):
    joint_idx = _joint_idx[joint_set_name]
    joints = [all_joint_names[j] for j in joint_idx]
    parent_idx = _parents[order][joint_set_name]
    parents = [joints[p] for p in parent_idx]
    n_joints = len(joint_idx)
    for i in range(n_joints):
        if parents[i] == joints[i]:
            parents[i] = None
        # print('%s -> %s' % (joints[i], parents[i]))
    return Skeleton(tuple(zip(joints, parents)))


base = skeleton('all', 'o1')
relevant = skeleton('relevant', 'o1')
extended = skeleton('extended', 'o1')
skeletons = {
    'base': base,
    'relevant': relevant,
    'extended': extended
}

converters = {
    'base': SkeletonConverter.identity,
    'relevant': SkeletonConverter(base, relevant),
    'extended': SkeletonConverter(base, extended)
}
