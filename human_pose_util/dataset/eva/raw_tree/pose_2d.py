import numpy as np
import cv2

import pose

# 2D pose
#
# The 2D pose converted from 3D pose using the calibration parameter


class Pose2D(pose.Pose):
    # constructor
    # @param x_3d The 3D pose
    # @param cam_param The camera calibration parameter
    def __init__(self, x_3d, cam_param):
        super(Pose2D, self).__init__()
        k, v = x_3d.get()
        x_2d = cv2.projectPoints(
            v, cam_param.r_c, cam_param.t_c, cam_param.A, cam_param.kc,
            cam_param.alpha_c)[0]
        for _k, _v in zip(k, x_2d):
            self._data[_k] = np.matrix(_v).T
    # modify the 2D pose according to bounding box
    # @param self The object pointer
    # @param bb The bounding box of human

    def modify(self, bb):
        for k in self._data.keys():
            self._data[k] = (self._data[k] - bb.u_0[0:2]) * bb.s
