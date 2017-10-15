from base import SkeletonConverter

_oi_map = {'head': 'head-back'}


def s24_to_s14_converter():
    from human_pose_util.dataset.h3m.skeleton import s24
    from human_pose_util.dataset.eva.skeleton import s14
    return SkeletonConverter(s24, s14, _oi_map)


def s24_to_s16_converter():
    from human_pose_util.dataset.h3m.skeleton import s24
    from human_pose_util.dataset.eva.skeleton import s16
    return SkeletonConverter(s24, s16, _oi_map)
