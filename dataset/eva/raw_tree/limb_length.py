import numpy as np

# Limb length
#
# The limb length of the current frame


class LimbLength(object):
    # constructor
    # @param mocap The motion capture data
    # @param frame The current frame
    def __init__(self, mocap, frame):
        self.__length = np.zeros(18)
        marker = mocap.marker[int(np.floor(frame)):int(np.ceil(frame)) + 1]

        def f(_from, _to): return np.mean(np.sqrt(
            ((marker[:, mocap.parameter[_from]] -
                marker[:, mocap.parameter[_to]])**2).sum(axis=1)))
        self.__length[0] = f("JOINT_CENTER_THORAX", "JOINT_CENTER_PELVIS")
        self.__length[1] = f("JOINT_CENTER_PELVIS", "JOINT_CENTER_LEFT_HIP")
        self.__length[2] = f("JOINT_CENTER_LEFT_HIP", "JOINT_CENTER_LEFT_KNEE")
        self.__length[3] = f("JOINT_CENTER_LEFT_KNEE",
                             "JOINT_CENTER_LEFT_ANKLE")
        self.__length[4] = 0
        self.__length[5] = f("JOINT_CENTER_PELVIS", "JOINT_CENTER_RIGHT_HIP")
        self.__length[6] = f("JOINT_CENTER_RIGHT_HIP",
                             "JOINT_CENTER_RIGHT_KNEE")
        self.__length[7] = f("JOINT_CENTER_RIGHT_KNEE",
                             "JOINT_CENTER_RIGHT_ANKLE")
        self.__length[8] = 0
        self.__length[9] = f("JOINT_CENTER_THORAX",
                             "JOINT_CENTER_LEFT_CLAVICLE")
        self.__length[10] = f("JOINT_CENTER_LEFT_SHOULDER",
                              "JOINT_CENTER_LEFT_ELBOW")
        self.__length[11] = f("JOINT_CENTER_LEFT_ELBOW",
                              "JOINT_CENTER_LEFT_WRIST")
        self.__length[12] = 0
        self.__length[13] = f("JOINT_CENTER_THORAX",
                              "JOINT_CENTER_RIGHT_CLAVICLE")
        self.__length[14] = f("JOINT_CENTER_RIGHT_SHOULDER",
                              "JOINT_CENTER_RIGHT_ELBOW")
        self.__length[15] = f("JOINT_CENTER_RIGHT_ELBOW",
                              "JOINT_CENTER_RIGHT_WRIST")
        self.__length[16] = 0
        self.__length[17] = 316.103
    # get length
    # @param self The object pointer
    # @param index The length index

    def __getitem__(self, index):
        return self.__length[index]
