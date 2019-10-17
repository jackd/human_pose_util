# import numpy as np
# from human_pose_util.register import dataset_register, skeleton_register
# from human_pose_util.evaluate import procrustes_error
# from human_pose_util.evaluate import sequence_procrustes_error
# from human_pose_util.skeleton import s24_to_s14_converter
from human_pose_util.dataset.report_manager import ReportManager
from human_pose_util.dataset.report_manager import proc_summary
from human_pose_util.dataset.report_manager import sequence_proc_summary
# from human_pose_util.dataset.h3m.skeleton import s24
# from human_pose_util.dataset.eva.skeleton import s16, s16_to_s14_converter


_categories = [
    'Direct', 'Discuss', 'Eat', 'Greet', 'Phone', 'Photo', 'Pose', 'Purchase',
    'Sit', 'SitDown', 'Smoke', 'Wait', 'Walk', 'WalkDog', 'WalkTogether']

_sub_categories = {
    'Direct': ['Directions', 'Directions 1'],
    'Discuss': ['Discussion 1', 'Discussion 2'],
    'Eat': ['Eating', 'Eating 1'],
    'Greet': ['Greeting', 'Greeting 1', 'Greeting 2'],
    'Phone': ['Phoning', 'Phoning 1', 'Phoning 2', 'Phoning 3'],
    'Photo': ['Photo', 'Photo 1'],
    'Pose': ['Posing', 'Posing 1'],
    'Purchase': ['Purchases', 'Purchases 1'],
    'Sit': ['Sitting', 'Sitting 1'],
    'SitDown': ['SittingDown', 'SittingDown 1'],
    'Smoke': ['Smoking', 'Smoking 1', 'Smoking 2'],
    'Wait': ['Waiting', 'Waiting 1'],
    'Walk': ['Walking', 'Walking 1'],
    'WalkDog': ['WalkDog', 'WalkDog 1'],
    'WalkTogether': ['WalkTogether', 'WalkTogether 1'],
}

_super_category = {}
for parent, children in _sub_categories.items():
    for child in children:
        _super_category[child] = parent


def category(example):
    return _super_category[example.attrs['sequence_id']]


def proc_manager():
    return ReportManager('proc_error', proc_summary, category)


def sequence_proc_manager():
    return ReportManager(
        'sequence_proc_error', sequence_proc_summary, category)
