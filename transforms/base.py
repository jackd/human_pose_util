from __future__ import division
from abc import abstractmethod


def _rotate_about_x(x, y, z, c, s):
    return x, c*y - s*z, s*y + c*z


def _rotate_about_y(x, y, z, c, s):
    return c*x + s*z, y, -s*x + c*z


def _rotate_about_z(x, y, z, c, s):
    return c*x - s*y, s*x + c*y, z


_rotate_about_fns = [_rotate_about_x, _rotate_about_y, _rotate_about_z]


_letters = 'abcdefghijklmnopqrstuv'

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

_NEXT_AXIS = [1, 2, 0, 1]


class Transform(object):
    """
    Base class for building high-level 3d transformations.

    Intended for a uniform numpy and tensorflow interface.
    """
    @abstractmethod
    def eps(self, dtype):
        pass

    @abstractmethod
    def sqrt(self, x):
        pass

    @abstractmethod
    def ternary(self, condition, if_true_fn, if_false_fn):
        pass

    @abstractmethod
    def stack(self, values, axis, **kwargs):
        pass

    @abstractmethod
    def unstack(self, value, num=None, axis=0, **kwargs):
        pass

    @abstractmethod
    def squeeze(self, input, axis=None, squeeze_dims=None, **kwargs):
        pass

    @abstractmethod
    def expand_dims(self, input, axis=None, **kwargs):
        pass

    @abstractmethod
    def cos(self, x, **kwargs):
        pass

    @abstractmethod
    def sin(self, x, **kwargs):
        pass

    @abstractmethod
    def tan(self, x, **kwargs):
        pass

    @abstractmethod
    def atan2(self, y, x, **kwargs):
        pass

    @abstractmethod
    def einsum(self, *args, **kwargs):
        pass

    @abstractmethod
    def reshape(self, tensor, shape, **kwargs):
        pass

    @abstractmethod
    def transpose(self, tensor, perm):
        pass

    @abstractmethod
    def split(self, value, num_or_size_splits, axis=0, **kwargs):
        pass

    @abstractmethod
    def reduce_sum(self, x, axis=None, keep_dims=False):
        pass

    def matmul(self, A, B, transpose_a=False, transpose_b=False):
        """
        Standard matrix multiplication.

        A and B are both 2D tensors.
        """
        return self.general_matmul(
            A, B, transpose_a=transpose_a, transpose_b=transpose_b)

    def batch_matmul(self, A, B, transpose_a=False, transpose_b=False):
        """
        Like matmul, but with additional dimension at front.

        Equivalent to:
        self.stack(
            [matmul(a, b, transpose_a, transpose_b) for (a, b) in zip(
             self.unstack(A, axis=0), self.unstack(B, axis=0))], axis=0)
        """
        # # The following implementation uses dimension broadcasting and sum_
        # # Not sure how to do the transpose_a and transpose_b case without a
        # # transpose. It does allow broadcasting on batch dims of A and B,
        # # unlike the einsum implementation below.
        if transpose_a:
            if transpose_b:
                print('WARNING: using inefficient batch matmul...')
                n = len(A.shape)
                return self.transpose(
                    self.batch_matmul(B, A), range(n-2) + [n-1, n-2])
                # A = self.expand_dims(A, -3)
                # B = self.expand_dims(B, -1)
                # return self.reduce_sum(A*B, axis=-2)
            else:
                A = self.expand_dims(A, axis=-1)
                B = self.expand_dims(B, axis=-2)
                return self.reduce_sum(A*B, axis=-3)
        elif transpose_b:
            A = self.expand_dims(A, axis=-2)
            B = self.expand_dims(B, axis=-3)
            return self.reduce_sum(A*B, axis=-1)
        else:
            A = self.expand_dims(A, axis=-1)
            B = self.expand_dims(B, axis=-3)
            return self.reduce_sum(A*B, axis=-2)

        # # Implementation that doesn't require a transpose when
        # # transpose_a and transpose_b
        # n_batch_dims = len(A.shape) - 2
        # common = _letters[:n_batch_dims]
        # a_str = common + ('xw' if transpose_a else 'wx')
        # b_str = common + ('yx' if transpose_b else 'xy')
        # equation = '%s,%s->%swy' % (a_str, b_str, common)
        # return self.einsum(equation, A, B)

    def general_matmul(self, A, B, transpose_a=False, transpose_b=False):
        """
        General matrix-matrix product.

        Like matmul, except A can have additional leading dimensions, and B
        can have additional trailing dimensions.
        """
        if len(A.shape) + len(B.shape) >= len(_letters):
            raise Exception('Too many dimensions!')
        a_str = _letters[:len(A.shape)-1]
        b_str = _letters[-1:-len(B.shape):-1]
        rhs_str = a_str + b_str

        a_str = a_str[:-1] + 'z' + a_str[-1] if transpose_a else a_str + 'z'
        b_str = b_str[0] + 'z' + b_str[1:] if transpose_b else 'z' + b_str
        return self.einsum('%s,%s->%s' % (a_str, b_str, rhs_str), A, B)

    def _stack_recursive(self, M, axis=0):
        if axis < 0:
            Ms = M
            while isinstance(Ms, (list, tuple)):
                Ms = Ms[0]
            axis += len(Ms.shape) + 1
        assert(axis >= 0)
        if isinstance(M, (list, tuple)):
            return self.stack(
                [self._stack_recursive(m, axis=axis) for m in M], axis=axis)
        else:
            return M

    def euler_matrix_nh(self, ai, aj, ak, axes='sxyz', stack_axis=-1):
        """
        Return homogeneous rotation matrix from Euler angles and axis sequence.

        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple

        >>> R = euler_matrix(1, 2, 3, 'syxz')
        >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
        True
        >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
        >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
        True
        >>> ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)
        >>> for axes in _TUPLE2AXES.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)

        """
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        if frame:
            ai, ak = ak, ai
        if parity:
            ai, aj, ak = -ai, -aj, -ak

        si, sj, sk = self.sin(ai), self.sin(aj), self.sin(ak)
        ci, cj, ck = self.cos(ai), self.cos(aj), self.cos(ak)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk

        M = [[None for ii in range(3)] for jj in range(3)]
        if repetition:
            M[i][i] = cj
            M[i][j] = sj*si
            M[i][k] = sj*ci
            M[j][i] = sj*sk
            M[j][j] = -cj*ss+cc
            M[j][k] = -cj*cs-sc
            M[k][i] = -sj*ck
            M[k][j] = cj*sc+cs
            M[k][k] = cj*cc-ss
        else:
            M[i][i] = cj*ck
            M[i][j] = sj*sc-cs
            M[i][k] = sj*cc+ss
            M[j][i] = cj*sk
            M[j][j] = sj*ss+cc
            M[j][k] = sj*cs-sc
            M[k][i] = -sj
            M[k][j] = cj*si
            M[k][k] = cj*ci
        M = self._stack_recursive(M, axis=stack_axis)
        return M

    def euler_from_matrix_nh(self, matrix, axes='sxyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix.

        >>> R0 = euler_matrix_nh(1, 2, 3, 'syxz')
        >>> al, be, ga = euler_from_matrix_nh(R0, 'syxz')
        >>> R1 = euler_matrix_nh(al, be, ga, 'syxz')
        >>> numpy.allclose(R0, R1)
        True
        >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R0 = euler_matrix(axes=axes, *angles)
        ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        ...    if not numpy.allclose(R0, R1): print(axes, "failed")

        """
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = matrix
        eps = self.eps(M.dtype)

        if repetition:
            sy = self.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            condition = sy > eps

            def if_true_fn():
                ax = self.atan2(M[i, j],  M[i, k])
                ay = self.atan2(sy,       M[i, i])
                az = self.atan2(M[j, i], -M[k, i])
                return self.stack([ax, ay, az], axis=-1)

            def if_false_fn():
                ax = self.atan2(-M[j, k],  M[j, j])
                ay = self.atan2(sy,       M[i, i])
                az = 0.0
                return self.stack([ax, ay, az], axis=-1)

            aaa = self.ternary(condition, if_true_fn, if_false_fn)

        else:
            cy = self.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            condition = cy > eps

            def if_true_fn():
                ax = self.atan2(M[k, j],  M[k, k])
                ay = self.atan2(-M[k, i],  cy)
                az = self.atan2(M[j, i],  M[i, i])
                return self.stack([ax, ay, az], axis=-1)

            def if_false_fn():
                ax = self.atan2(-M[j, k],  M[j, j])
                ay = self.atan2(-M[k, i],  cy)
                az = 0.0
                return self.stack([ax, ay, az], axis=-1)

            aaa = self.ternary(condition, if_true_fn, if_false_fn)

        if parity:
            aaa *= -1
        ax, ay, az = self.unstack(aaa, axis=-1)
        if frame:
            ax, az = az, ax
        return ax, ay, az

    def rotation_matrix(self, r, stack_axis=0):
        """Get the rotation matrix associated with euler angles r."""
        ai, aj, ak = self.unstack(r, axis=-1)
        return self.euler_matrix_nh(ai, aj, ak, stack_axis=stack_axis)

    def rotation_with_matrix(self, points, R, inverse=False):
        """
        Rotate points by rotation matrix R.

        Performs inverse rotation if inverse is True.
        """
        assert(R.shape[-2:] == (3, 3))
        if len(R.shape) == 2 and len(points.shape) == 3:
            R = self.expand_dims(R, axis=0)
        assert(len(R.shape) == len(points.shape))
        return self.batch_matmul(points, R, transpose_b=not inverse)

    def rotate(self, points, r, inverse=False):
        """
        Rotate points by euler-angles r.

        args:
            points: (N, n_joints, 3) or (n_joints, 3) array
            r: (N, 3) or (3,) array, euler angles, [ai, aj, ak], in radians
            inverse: if true, performs inverse rotation.
        Returns:
            rotated points.
        """
        return self.rotation_with_matrix(
            points, self.rotation_matrix(r, stack_axis=-1), inverse=inverse)

    def rotate_about(self, points, angle, dim, inverse=False, axis=-1):
        """Optimized rotate for single-axis rotation."""
        if inverse:
            angle = -angle
        c = self.cos(angle)
        s = self.sin(angle)
        x, y, z = self.unstack(points, axis=axis)
        x, y, z = _rotate_about_fns[dim](x, y, z, c, s)
        return self.stack([x, y, z], axis=axis)

    def transform_frame(self, points, r, t, inverse=False):
        """Combines rotation in euler-angles (r, xyz) with translation (t)."""
        ps = points.shape
        ts = t.shape
        if len(ts) == len(ps) - 1 and ts[:-1] == ps[:-2] and ts[-1] == ps[-1]:
            t = self.expand_dims(t, axis=-2)
        if inverse:
            return self.rotate(points - t, r, inverse=True)
        else:
            return self.rotate(points, r, inverse=False) + t

    def project(self, points, f, c, axis=-1):
        """Project 3d points to 2d using camera focal length/pixel offset."""
        assert(points.shape[-1] == 3)
        xy, z = self.split(points, [2, 1], axis=-1)
        return xy/z * f + c
