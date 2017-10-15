import numpy as np
import scipy.io

# Mocap data
#
# The motion capture data


class MotionCaptureData(object):
    # define const values
    MARKERS = ("RASI", "LASI", "LPSI", "RELB", "STRN", "LTHI", "LWRA", "LELB",
               "T10", "RWRA", "RUPA", "RKNE", "RBAK", "LKNE", "CLAV", "RSHO",
               "LSHO", "C7", "RTIB", "LTIB", "LBHD", "RBHD", "RFHD", "LFHD",
               "LANK", "LHEE", "RANK", "LTOE", "RTOE", "RHEE", "LFIN",
               ("RFRM", "RFRA"), ("LFRM", "LFRA"), "RWRB", "RFIN", "LWRB",
               "RTHI", "LUPA", "RPSI")
    # constructor
    # @param mat_file The filename of the mocap data

    def __init__(self, mat_file):
        mat_data = scipy.io.loadmat(mat_file)
        # marker data
        self.marker = mat_data["Markers"]
        # load marker data
        param_group = mat_data["ParameterGroup"]
        group_id = [x[0, 0] for x in param_group[0]["name"]].index("POINT")
        sub_group_id = [x[0, 0] for x in param_group[0]
                        [group_id]["Parameter"]["name"][0]].index("LABELS")
        self.__data = param_group[
            0][group_id]["Parameter"][0][sub_group_id]["data"][0]
        # mocap parameter
        self.parameter = {"COORD_PELVIS": "PELO",
                          "COORD_THORAX": "TRXO",
                          "COORD_HEAD": "HEDO",
                          "COORD_LEFT_FEMUR": "LFEO",
                          "COORD_LEFT_TIBIA": "LTIO",
                          "COORD_LEFT_FOOT": "LFOO",
                          "COORD_LEFT_TOE": "LTOO",
                          "COORD_RIGHT_FEMUR": "RFEO",
                          "COORD_RIGHT_TIBIA": "RTIO",
                          "COORD_RIGHT_FOOT": "RFOO",
                          "COORD_RIGHT_TOE": "RTOO",
                          "COORD_LEFT_CLAVICLE": "LCLO",
                          "COORD_LEFT_UPPER_ARM": "LHUO",
                          "COORD_LEFT_LOWER_ARM": "LRAO",
                          "COORD_LEFT_HAND": "LHNO",
                          "COORD_RIGHT_CLAVICLE": "RCLO",
                          "COORD_RIGHT_UPPER_ARM": "RHUO",
                          "COORD_RIGHT_LOWER_ARM": "RRAO",
                          "COORD_RIGHT_HAND": "RHNO",
                          "JOINT_CENTER_PELVIS": "PELO",
                          "JOINT_CENTER_THORAX": "TRXO",
                          "JOINT_CENTER_HEAD": "TRXO",
                          "JOINT_CENTER_LEFT_HIP": "LFEP",
                          "JOINT_CENTER_LEFT_KNEE": "LFEO",
                          "JOINT_CENTER_LEFT_ANKLE": "LTIO",
                          "JOINT_CENTER_RIGHT_HIP": "RFEP",
                          "JOINT_CENTER_RIGHT_KNEE": "RFEO",
                          "JOINT_CENTER_RIGHT_ANKLE": "RTIO",
                          "JOINT_CENTER_LEFT_CLAVICLE": "LCLO",
                          "JOINT_CENTER_LEFT_SHOULDER": "LCLO",
                          "JOINT_CENTER_LEFT_ELBOW": "LHUO",
                          "JOINT_CENTER_LEFT_WRIST": "LRAO",
                          "JOINT_CENTER_RIGHT_CLAVICLE": "RCLO",
                          "JOINT_CENTER_RIGHT_SHOULDER": "RCLO",
                          "JOINT_CENTER_RIGHT_ELBOW": "RHUO",
                          "JOINT_CENTER_RIGHT_WRIST": "RRAO",
                          "MARKER_LSHO": "LSHO",
                          "MARKER_RSHO": "RSHO",
                          "MARKER_STRN": "STRN",
                          "MARKER_T10": "T10",
                          "MARKER_LBHD": "LBHD",
                          "MARKER_LFHD": "LFHD",
                          "MARKER_RFHD": "RFHD",
                          "MARKER_RBHD": "RBHD"}
        for k, v in self.parameter.items():
            try:
                self.parameter[k] = [i for i, x in enumerate(
                    self.__data) if v in x[0]][0]
            except IndexError:
                raise RuntimeError(
                    "The motion capture data file ({0}) is not valid.".format(
                        mat_file))
    # check whether all markers is visible
    # @param self The object pointer
    # @param frame The current frame (type:float)

    def isValid(self, frame):
        for marker in self.MARKERS:
            if isinstance(marker, str):
                index = [i for i, x in enumerate(
                    self.__data) if marker in x[0]][0]
            else:
                index = [i for i, x in enumerate(self.__data) if any(
                    [m in x[0] for m in marker])][0]
            if (self.marker[int(np.floor(frame)), index] == 0).all() or (
                    self.marker[int(np.ceil(frame)), index] == 0).all():
                return False
        return True
