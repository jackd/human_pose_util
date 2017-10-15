from human_pose_util.dataset.report_manager import ReportManager
from human_pose_util.dataset.report_manager import proc_summary
from human_pose_util.dataset.report_manager import sequence_proc_summary


def category(example):
    return example.attrs['base_sequence']


def proc_manager():
    return ReportManager('proc_error', proc_summary, category)


def sequence_proc_manager():
    return ReportManager(
        'sequence_proc_error', sequence_proc_summary, category)
