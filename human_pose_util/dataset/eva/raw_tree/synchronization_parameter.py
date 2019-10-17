## Synchronization parameter
#
# The parameter of synchronizing data between the image and the motion capture streams
class SynchronizationParameter(object):
    ## constructor
    # @param ofs_file The filename of the synchronization data
    def __init__(self, ofs_file):
        f = open(ofs_file)
        ## starting frame index in the image stream
        self.im_st = float(f.readline())
        ## corresponding frame index in mocap stream
        self.mc_st = float(f.readline())
        ## temporal scaling between streams
        self.mc_sc = float(f.readline())
        f.close()
