# Actor parameter
#
# The measurement parameters of actor


class ActorParameter(object):
    # constructor
    # @param mp_file The filename of the actor measurement data
    def __init__(self, mp_file):
        for line in open(mp_file):
            splited = line.split(" ")
            setattr(self, splited[0][1:], float(splited[2]))
