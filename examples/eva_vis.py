from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from human_pose_util.register import register_datasets, get_dataset, get_skeleton, get_converter

register_datasets(eva=True)
eva = get_dataset('eva')
key = eva.keys()[0]
p3w = np.array(eva[key]['p3w'])
p3w /= 1000
target = 's14'
converter = get_converter('s20', target)
skeleton = get_skeleton(target)
p3w = converter.convert(p3w)

# matplotlib vis
from human_pose_util.skeleton.vis import vis3d
import matplotlib.pyplot as plt
vis3d(skeleton, p3w[0])
plt.show()

# animation vis with glumpy
from human_pose_util.animation import animated_scene as anim
anim.add_limb_collection_animator(skeleton, p3w, 60)
anim.run(60)
