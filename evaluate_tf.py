from __future__ import print_function
import numpy as np
import tensorflow as tf
from transforms.tf_impl import rotate


def reconstruction_error_tf(ground_truth, inferred):
    """Tensorflow implementation of `reconstruction_error`."""
    return tf.sqrt(tf.reduce_sum((inferred - ground_truth)**2, axis=-1))
    # return tf.reduce_sum((inferred - ground_truth)**2, axis=-1)


class ReconstructionTransformOptimizer(object):
    """
    Optimizer for finding the transform which minimizes recon. loss.

    procruste analysis minimizes the 2-norm. This will be slower, but will
    minimize the appropriate function.

    Implementation is based on `tf.contrib.opt.ScipyOptimizerInterface`
    """
    def __init__(self, n_joints, max_frames, **optimizer_kwargs):
        # self._ground_truth = tf.placeholder(
        #     shape=(None, 3), dtype=tf.float32, name='ground_truth')
        # self._inferred = tf.placeholder(
        #     shape=(None, 3), dtype=tf.float32, name='inferred')
        self._max_frames = max_frames
        self._n_joints = n_joints
        self._n = tf.Variable(0, dtype=np.int32)
        gt, inf = self.null_vals()
        self._ground_truth = tf.Variable(gt, dtype=tf.float32)
        self._inferred = tf.Variable(inf, dtype=tf.float32)
        self._r = tf.Variable(
            np.zeros((max_frames, 3)), dtype=tf.float32, name='r')
        self._scale = tf.Variable(
            np.ones((max_frames,)), dtype=tf.float32, name='scale')
        scale = tf.expand_dims(tf.expand_dims(
            self._scale[:self._n], axis=-1), axis=-1)
        self._transformed = scale*rotate(
            self._inferred[:self._n],
            # tf.expand_dims(self._r, axis=1)
            self._r[:self._n])
        self._err = reconstruction_error_tf(
            self._ground_truth[:self._n],
            self._transformed)
        self._loss = tf.reduce_sum(self._err)
        self._opt_vars = [self._r, self._scale]
        self.optimizer_kwargs = optimizer_kwargs
        self.open()

    def null_vals(self):
        gt = np.zeros((self._max_frames, self._n_joints, 3), dtype=np.float32)
        inf = np.zeros((self._max_frames, self._n_joints, 3), dtype=np.float32)
        return gt, inf

    def _optimize(self, ground_truth, inferred, **run_kwargs):
        assert(len(ground_truth.shape) == 3)
        assert(ground_truth.shape[-1] == 3)
        assert(ground_truth.shape == inferred.shape)
        n = ground_truth.shape[0]
        assert(n <= self._max_frames)
        ground_truth = ground_truth - np.expand_dims(np.mean(
            ground_truth, axis=-2), axis=-2)
        inferred = inferred - np.expand_dims(np.mean(
            inferred, axis=-2), axis=-2)
        gt, inf = self.null_vals()
        gt[:n] = ground_truth
        inf[:n] = inferred
        self._sess.run(tf.variables_initializer(self._opt_vars))
        self._sess.run(
            [self._ground_truth.assign(gt),
             self._inferred.assign(inf),
             self._n.assign(n)])
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self._loss, self._opt_vars, **self.optimizer_kwargs)
        optimizer.minimize(self._sess, **run_kwargs)
        return self._sess.run(self._err)

    def optimize(self, ground_truth, inferred, **run_kwargs):
        """
        Find optimal transform.

        Returns:
            optimal_loss
            recentered ground_truth
            transformed inferred
        """
        assert(ground_truth.shape[-1] == 3)
        assert(ground_truth.shape == inferred.shape)
        if len(ground_truth.shape) == 2:
            return self._optimize(
                np.expand_dims(ground_truth, axis=0),
                np.expand_dims(inferred, axis=0), **run_kwargs)[0]
        elif len(ground_truth.shape) == 3:
            return self._optimize(ground_truth, inferred, **run_kwargs)
        else:
            raise ValueError('ground_truth must have 2 or 3 dims.')

    def open(self):
        self._sess = tf.Session()

    def close(self):
        self._sess.close()
