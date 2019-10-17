"""
Module issues if done in `if __name__ == '__main'__` block of register.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from human_pose_util.register import register_skeletons, register_datasets
from human_pose_util.register import get_dataset
from human_pose_util.register import get_skeleton
from human_pose_util.dataset.normalize import normalized_view_data, normalized_p3w
from human_pose_util.skeleton.vis import vis3d

register_skeletons(h3m=True, eva=True, mpi_inf=True)
register_datasets(h3m=True, eva=True)
# register_converters(h3m_eva=True)
print('Registration successful!')

# dataset = dataset_register['h3m']

for dataset_id, target_skeleton_id in [['h3m', 's24'], ['eva', 's14']]:
    dataset = get_dataset(dataset_id)
    for mode in ['eval', 'train']:
        print('Getting normalized_view_data...')
        normalized_view_data(dataset, modes=mode)

        print('Getting normalized_p3w...')
        normalized_dataset, p3w = normalized_p3w(dataset,
                                                 modes=mode,
                                                 skeleton_id=target_skeleton_id)

skeleton = get_skeleton(normalized_dataset.attrs['skeleton_id'])
print(p3w.shape)
vis3d(skeleton, p3w[0])
plt.show()
