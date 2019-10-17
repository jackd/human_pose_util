import numpy as np

## pose
#
# The basic class of 2D/3D pose
class Pose(object):
    ## constructor
    def __init__(self):
        ## internal data expression
        self._data = {}
    ## get all poses
    # @param self The object pointer
    # @return joint_names, joint_pose
    def get(self):
        return self._data.keys(), np.array(self._data.values())
    ## get pose by name
    # @param self The object pointer
    # @param name The joint name
    def __getitem__(self, name):
        return self._data[name]
