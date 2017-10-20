from human_pose_util.transforms import np_impl
raise Exception('Deprecated')  # just to remind myself. Might want it later...


class PoseSequence(object):
    def __init__(
            self, p3w, r, t, f, c, subject_id, sequence_id, camera_id,
            skeleton):
        self._p3w = p3w
        self._r = r
        self._t = t
        self._f = f
        self._c = c
        self._subject_id = subject_id
        self._sequence_id = sequence_id
        self._camera_id = camera_id
        self._skeleton = skeleton
        self.pixel_scale = 1
        self.space_scale = 1

    @property
    def p3c(self):
        return np_impl.transform_frame(self.p3w, r=self.r, t=self.t)

    @property
    def p2(self):
        return np_impl.transform_frame(
            self.p3c, f=self.f, c=self.c) / self.pixel_scale

    @property
    def p3w(self):
        return self._p3w / self.space_scale

    @property
    def r(self):
        return self._r

    @property
    def t(self):
        return self._t / self.space_scale

    @property
    def f(self):
        return self._f / self.pixel_scale

    @property
    def c(self):
        return self._f / self.pixel_scale


class Hdf5PoseSequence(PoseSequence):
    def __init__(self, group):
        self._group = group

    @property
    def _p3w(self):
        return self._group['p3w']

    @property
    def _f(self):
        return self._group.attrs['f']

    @property
    def _c(self):
        return self._group.attrs['c']

    @property
    def _r(self):
        return self._group.attrs['r']

    @property
    def _t(self):
        return self._group.attrs['t']


if __name__ == '__main__':
    base = PoseSequence(0, 0, 0, 0, 0)
    print(base.p3w)
