import numpy as np
import cv2

# Camera parameter
#
# The structure of the camera parameter


class CameraParameter(object):
    # constructor
    # @param cal_file The filename of the calibration data
    def __init__(self, cal_file):
        # load camera parameter from calibration data
        f = open(cal_file)
        # focal length (2)
        fc = np.ones(2)
        for i in range(2):
            fc[i] = float(f.readline())
        # principle point (2x1)
        cc = np.ones((2, 1))
        for i in range(2):
            cc[i] = float(f.readline())
        # camera matrix
        self.A = np.matrix(np.identity(3))
        self.A[0:2, 0:2] = np.diag(fc)
        self.A[0:2, 2] = cc
        # inverse camera matrix
        self.A_inv = np.linalg.inv(self.A)
        # skew coefficient (1)
        self.alpha_c = 0
        self.alpha_c = float(f.readline())
        # distortion coefficient (5x1)
        self.kc = np.ones((5, 1))
        for i in range(5):
            self.kc[i] = float(f.readline())
        # extrinsic rotation matrix (3x3)
        self.Rc = np.identity(3)
        for i in range(9):
            self.Rc[i / 3, i % 3] = float(f.readline())
        # extrinsic rotation vector (3x1)
        self.r_c = cv2.Rodrigues(self.Rc)[0]
        # extrinsic translation vector (3x1)
        self.t_c = np.zeros((3, 1))
        for i in range(3):
            self.t_c[i] = float(f.readline())
        # release memory
        self.fc = fc
        self.cc = cc
        f.close()
