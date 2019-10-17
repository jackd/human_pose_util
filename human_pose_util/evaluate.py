import numpy as np
import scipy.spatial


def reconstruction_error(ground_truth, inferred):
    """|ground_truth - inferred|_2."""
    return np.sqrt(np.sum((inferred - ground_truth)**2, axis=-1))


def _procrustes(ground_truth, inferred):
    nj = ground_truth.shape[0]
    invalid = np.logical_or(
        np.any(np.isnan(ground_truth), axis=-1),
        np.any(np.isnan(inferred), axis=-1))

    valid = np.logical_not(invalid)
    ground_truth = ground_truth[valid]
    inferred = inferred[valid]

    if len(ground_truth) < 2:
        return np.nan
    gt = ground_truth - np.mean(ground_truth, 0)
    norm = np.linalg.norm(gt)
    assert(norm != 0)
    p3, p3_i, l2 = scipy.spatial.procrustes(gt, inferred)
    p3_i *= norm
    p3 *= norm
    err = reconstruction_error(p3, p3_i)

    ret_p3 = np.empty((nj, 3))
    ret_p3_i = np.empty((nj, 3))
    ret_err = np.empty((nj,))

    for ret in [ret_p3, ret_p3_i, ret_err]:
        ret[invalid] = np.nan

    ret_p3[valid] = p3
    ret_p3_i[valid] = p3_i
    ret_err[valid] = err

    return ret_err, ret_p3, ret_p3_i


def procrustes(ground_truth, inferred):
    """
    Get the procruste error between ground_truth and inferred.

    The Procruste error is the minimum reconstruction loss over all rigid
    transformations T of (T(inferred) - ground_truth).

    Arguments:
        ground_truth: data label, shape (n, m, 3) (batched) or (m, 3)
        inferred: inferred value, same shape as ground truth

    Returns:
        error: error value, or array of length n if batched
        p3: transformed ground_truth
        p3i: transformed inference
    """
    if ground_truth.shape != inferred.shape:
        raise Exception(
            'Shapes must be same but got %s, %s' %
            (str(ground_truth.shape), str(inferred.shape)))
    if len(ground_truth.shape) > 2:
        orig_shape = ground_truth.shape
        ground_truth_flat = np.reshape(ground_truth, (-1,) + orig_shape[-2:])
        inferred_flat = np.reshape(inferred, (-1,) + orig_shape[-2:])

        p3 = np.empty_like(inferred_flat)
        p3_i = np.empty_like(inferred_flat)
        err = np.empty(inferred_flat.shape[:-1], dtype=p3_i.dtype)
        for i, (gt, inf) in enumerate(zip(ground_truth_flat, inferred_flat)):
            err[i], p3[i], p3_i[i] = _procrustes(gt, inf)
        err = np.reshape(err, orig_shape[:-1])
        p3 = np.reshape(p3, orig_shape)
        p3_i = np.reshape(p3_i, orig_shape)
        return err, p3, p3_i
    else:
        return _procrustes(ground_truth, inferred)


def procrustes_error(ground_truth, inferred):
    """Get the error component of `procrustes`."""
    return procrustes(ground_truth, inferred)[0]


def sequence_procrustes(ground_truth, inferred):
    """
    Get the procruste error for a sequence.

    Uses the same optimal transform for all poses.

    Args:
        ground_truth: (n, m, 3)
        inferred: (n, m, 3)

    Returns:
        error: (n,)
        p3: (n, m, 3)
        p3i: (n, m, 3)
    """
    n, m = ground_truth.shape[:2]
    assert(ground_truth.shape == (n, m, 3))
    assert(ground_truth.shape == inferred.shape)
    err, p3, p3i = procrustes(
        ground_truth.reshape(-1, 3), inferred.reshape(-1, 3))
    p3 = p3.reshape(n, m, 3)
    p3i = p3i.reshape(n, m, 3)
    err = reconstruction_error(p3, p3i)
    return err, p3, p3i


def sequence_procrustes_error(ground_truth, inferred):
    """Get the error component of `sequence_procrustes`."""
    return sequence_procrustes(ground_truth, inferred)[0]
