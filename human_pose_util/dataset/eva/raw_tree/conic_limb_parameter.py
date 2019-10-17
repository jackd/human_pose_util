import numpy as np

## Conic limb parameter
#
# The conic parameters for each limb in the body model
# The params are defined as follows:
#
#                    +------------+---------------+-----------+-----------+                    
#                    | top radius | bottom radius | x-scaling | y-scaling |                     
#  +-----------------+------------+---------------+-----------+-----------+
#  | torso           |
#  | left thigh      |
#  | left calf       |
#  | right thigh     |
#  | right calf      |                      10 x 5 matrix 
#  | left upper arm  |
#  | left lower arm  |
#  | right upper arm |
#  | right lower arm |
#  | head            |
#  +-----------------+----------------------------------------------------+

class ConicLimbParameter(object):
    # define const values
    MARKER_SIZE = 14
    ## constructor
    # @param mocap The mocap data
    # @param actor_param The parameters of actor
    def __init__(self, mocap, actor_param):
        ## conic parameter
        self.__parameter = np.zeros((10, 5))
        # generate conic parameter from mocap data and limb measurement
        # toroso
        self.__parameter[0, 0] = np.mean(np.sqrt(((mocap.marker[:, mocap.parameter["MARKER_LSHO"]] - mocap.marker[:, mocap.parameter["MARKER_RSHO"]])**2).sum(axis=1)))/2 - self.MARKER_SIZE/2
        self.__parameter[0, 1] = self.__parameter[0, 0]
        self.__parameter[0, 2] = np.mean(np.sqrt(((mocap.marker[:, mocap.parameter["MARKER_STRN"]] - mocap.marker[:, mocap.parameter["MARKER_T10"]])**2).sum(axis=1)))/(2*self.__parameter[0, 0])
        self.__parameter[0, 3] = 1
        # left thigh
        self.__parameter[1, 0] = (self.__parameter[0, 0] + self.MARKER_SIZE/2)/2
        self.__parameter[1, 1] = actor_param.LKneeWidth/2
        self.__parameter[1, 2] = 1
        self.__parameter[1, 3] = 1
        # left calf
        self.__parameter[2, 0] = actor_param.LKneeWidth/2
        self.__parameter[2, 1] = actor_param.LAnkleWidth/2
        self.__parameter[2, 2] = 1
        self.__parameter[2, 3] = 1
        # right thigh
        self.__parameter[3, 0] = (self.__parameter[0, 0] + self.MARKER_SIZE/2)/2
        self.__parameter[3, 1] = actor_param.RKneeWidth/2
        self.__parameter[3, 2] = 1
        self.__parameter[3, 3] = 1
        # right calf
        self.__parameter[4, 0] = actor_param.RKneeWidth/2
        self.__parameter[4, 1] = actor_param.RAnkleWidth/2
        self.__parameter[4, 2] = 1
        self.__parameter[4, 3] = 1
        # left upper arm
        self.__parameter[5, 0] = actor_param.LShoulderOffset
        self.__parameter[5, 1] = actor_param.LElbowWidth/2
        self.__parameter[5, 2] = 1
        self.__parameter[5, 3] = 1
        # left lower arm
        self.__parameter[6, 0] = actor_param.LElbowWidth/2
        self.__parameter[6, 1] = actor_param.LWristWidth/2
        self.__parameter[6, 2] = 1
        self.__parameter[6, 3] = 1
        # right upper arm
        self.__parameter[7, 0] = actor_param.RShoulderOffset
        self.__parameter[7, 1] = actor_param.RElbowWidth/2
        self.__parameter[7, 2] = 1
        self.__parameter[7, 3] = 1
        # right lower arm
        self.__parameter[8, 0] = actor_param.RElbowWidth/2
        self.__parameter[8, 1] = actor_param.RWristWidth/2
        self.__parameter[8, 2] = 1
        self.__parameter[8, 3] = 1
        # head
        self.__parameter[9, 0] = np.mean(np.sqrt(((mocap.marker[:, mocap.parameter["MARKER_LBHD"]] - mocap.marker[:, mocap.parameter["MARKER_RFHD"]])**2).sum(axis=1)))/2
        self.__parameter[9, 1] = self.__parameter[9, 0]
        self.__parameter[9, 2] = 1
        self.__parameter[9, 3] = np.mean(np.sqrt(((mocap.marker[:, mocap.parameter["MARKER_LFHD"]] - mocap.marker[:, mocap.parameter["MARKER_RFHD"]])**2).sum(axis=1)))/(2*self.__parameter[9, 0]) # TODO:check the usage of y-scaling. RFHD->RBHD?
        self.__parameter[9, 4] = actor_param.HeadOffset/180*np.pi
    ## get parameter
    # @param self The object pointer
    # @param index The parameter index
    def __getitem__(self, index):
        return self.__parameter[index]
