import numpy as np
import pyquaternion as quat

# Global marker transform
#
# The global transform of markers


class GlobalMarkerTransform(object):
    # constructor
    # @param mocap The motion capture data
    # @param frame The current frame
    # @param conic_param The conic limb parameter
    # @param length The limb length
    def __init__(self, mocap, frame, conic_param, length):
        # body
        self.pelvis = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_PELVIS"],
            mocap.parameter["JOINT_CENTER_PELVIS"])
        self.lpelvis = self.pelvis * \
            quat.Quaternion(
                axis=[0, 1, 0], radians=np.pi).transformation_matrix
        self.rpelvis = self.lpelvis
        self.thorax = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_THORAX"],
            mocap.parameter["JOINT_CENTER_THORAX"])
        self.lthorax = self.thorax * \
            quat.Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2).transformation_matrix
        self.rthorax = self.thorax * \
            quat.Quaternion(axis=[1, 0, 0], radians=-
                            np.pi / 2).transformation_matrix
        self.head = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_HEAD"],
            mocap.parameter["JOINT_CENTER_HEAD"])
        # uppper body
        self.lclavicle = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_LEFT_CLAVICLE"],
            mocap.parameter["JOINT_CENTER_LEFT_CLAVICLE"])
        self.lshoulder = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_LEFT_UPPER_ARM"],
            mocap.parameter["JOINT_CENTER_LEFT_SHOULDER"])
        self.lelbow = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_LEFT_LOWER_ARM"],
            mocap.parameter["JOINT_CENTER_LEFT_ELBOW"])
        self.rclavicle = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_RIGHT_CLAVICLE"],
            mocap.parameter["JOINT_CENTER_RIGHT_CLAVICLE"])
        self.rshoulder = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_RIGHT_UPPER_ARM"],
            mocap.parameter["JOINT_CENTER_RIGHT_SHOULDER"])
        self.relbow = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_RIGHT_LOWER_ARM"],
            mocap.parameter["JOINT_CENTER_RIGHT_ELBOW"])
        # lower body
        self.lthigh = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_LEFT_FEMUR"],
            mocap.parameter["JOINT_CENTER_LEFT_HIP"], quat.Quaternion(
                axis=[0, 1, 0], radians=np.pi).transformation_matrix)
        self.ltibia = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_LEFT_TIBIA"],
            mocap.parameter["JOINT_CENTER_LEFT_KNEE"], quat.Quaternion(
                axis=[0, 1, 0], radians=np.pi).transformation_matrix)
        self.rthigh = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_RIGHT_FEMUR"],
            mocap.parameter["JOINT_CENTER_RIGHT_HIP"], quat.Quaternion(
                axis=[0, 1, 0], radians=np.pi).transformation_matrix)
        self.rtibia = self.__computeGlobalTransform(
            mocap, frame, mocap.parameter["COORD_RIGHT_TIBIA"],
            mocap.parameter["JOINT_CENTER_RIGHT_KNEE"], quat.Quaternion(
                axis=[0, 1, 0], radians=np.pi).transformation_matrix)
        # modify pose according to body local coordinate
        self.pelvis = self.pelvis * np.vstack(
            (np.hstack((np.identity(3), np.matrix([-50, 0, 0]).T)), np.matrix(
                [0, 0, 0, 1]))) * quat.Quaternion(
                    axis=[0, 1, 0], radians=np.arctan2(
                        conic_param[0, 0] / 2,
                        length[0])).transformation_matrix
        diff = self.lelbow[0:3, 3] - \
            (self.lshoulder[0:3] * np.matrix([0, 0, -length[10], 1]).T)
        self.lshoulder = self.lshoulder * quat.Quaternion(
            axis=[0, 1, 0], radians=-np.arctan2(
                conic_param[5, 0] / 2, length[10])).transformation_matrix
        self.lelbow[0:3, 3] = self.lshoulder[0:3] * \
            np.matrix([0, 0, -length[10], 1]).T + diff
        self.lelbow = self.lelbow * quat.Quaternion(
            axis=[0, 1, 0], radians=np.arctan2(
                conic_param[5, 0] / 2, length[10])).transformation_matrix
        diff = self.relbow[0:3, 3] - \
            (self.rshoulder[0:3] * np.matrix([0, 0, -length[14], 1]).T)
        self.rshoulder = self.rshoulder * quat.Quaternion(
            axis=[0, 1, 0], radians=-np.arctan2(
                conic_param[7, 0] / 2, length[14])).transformation_matrix
        self.relbow[0:3, 3] = self.rshoulder[0:3] * \
            np.matrix([0, 0, -length[14], 1]).T + diff
        self.relbow = self.relbow * quat.Quaternion(
            axis=[0, 1, 0], radians=np.arctan2(
                conic_param[7, 0] / 2, length[14])).transformation_matrix
    # compute global transform
    # @param self The object pointer
    # @param mocap The motion capture data
    # @param frame The current frame
    # @param coord The coordinate ID of the transform
    # @param origin The origin ID of the transform
    # @param rot_offset The rotation offset of the transform

    def __computeGlobalTransform(
            self, mocap, frame, coord, origin, rot_offset=np.identity(4)):
        frame_index = int(np.floor(frame))
        frame_index_next = frame_index + 1
        if frame_index_next >= mocap.marker.shape[0]:
            return np.matrix(np.ones((4, 4))*np.nan)
        # get marker position of current frame index and next frame index
        origin_location_f = np.matrix(mocap.marker[frame_index, origin])
        origin_location_c = np.matrix(mocap.marker[frame_index_next, origin])
        oaxis_point_f = np.matrix(mocap.marker[frame_index, coord])
        oaxis_point_c = np.matrix(mocap.marker[frame_index_next, coord])
        xaxis_point_f = np.matrix(mocap.marker[frame_index, coord + 2])
        xaxis_point_c = np.matrix(mocap.marker[frame_index_next, coord + 2])
        yaxis_point_f = np.matrix(mocap.marker[frame_index, coord + 3])
        yaxis_point_c = np.matrix(mocap.marker[frame_index_next, coord + 3])
        zaxis_point_f = np.matrix(mocap.marker[frame_index, coord + 1])
        zaxis_point_c = np.matrix(mocap.marker[frame_index_next, coord + 1])
        # compute global coordinate from base markers

        def n(v):
            norm = np.linalg.norm(v)
            return v if norm == 0 else v / np.linalg.norm(v)

        xaxis_vect_f = n(xaxis_point_f - oaxis_point_f)
        yaxis_vect_f = n(yaxis_point_f - oaxis_point_f)
        zaxis_vect_f = n(zaxis_point_f - oaxis_point_f)
        xaxis_vect_c = n(xaxis_point_c - oaxis_point_c)
        yaxis_vect_c = n(yaxis_point_c - oaxis_point_c)
        zaxis_vect_c = n(zaxis_point_c - oaxis_point_c)
        localROTglobal_f = np.vstack(
            (xaxis_vect_f, yaxis_vect_f, zaxis_vect_f))
        localROTglobal_c = np.vstack(
            (xaxis_vect_c, yaxis_vect_c, zaxis_vect_c))
        # strict orthonormalization
        localROTglobal_f_orth = np.linalg.qr(localROTglobal_f)[0]
        localROTglobal_c_orth = np.linalg.qr(localROTglobal_c)[0]
        for i in range(3):
            e = localROTglobal_f[:, i] - localROTglobal_f_orth[:, i]
            if e.T * e > 1:
                localROTglobal_f_orth[:, i] *= -1
            e = localROTglobal_c[:, i] - localROTglobal_c_orth[:, i]
            if e.T * e > 1:
                localROTglobal_c_orth[:, i] *= -1
        if (np.linalg.norm(localROTglobal_f - localROTglobal_f_orth) > 1.e-5 or
                np.linalg.norm(
                    localROTglobal_c - localROTglobal_c_orth) > 1.e-5):
            return np.matrix(np.ones((4, 4))*np.nan)
            # coord_name = [x for x in [
            #     k for k, v in mocap.parameter.items() if v == coord]
            #         if "COORD" in x][0]
            # raise RuntimeError(
            #     "Bad marker data for '{0}' coordinate.".format(coord_name))
        # calculate current global coordinate interpolating f/c coordinate
        alpha = frame - frame_index
        localROTglobal = quat.Quaternion.slerp(
            quat.Quaternion(matrix=localROTglobal_f_orth), quat.Quaternion(
                matrix=localROTglobal_c_orth), alpha).transformation_matrix
        globalTRANSlocal = np.vstack(
            (np.hstack((np.identity(3), (1 - alpha) * origin_location_f.T
             + alpha * origin_location_c.T)), np.array([0, 0, 0, 1])))
        return globalTRANSlocal * np.linalg.inv(localROTglobal) * rot_offset
        # out = globalTRANSlocal * np.linalg.inv(localROTglobal) * rot_offset
        # print(out.shape)
        # print(out.dtype)
        # return out
