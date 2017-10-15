import tensorflow as tf


def examples_to_tf(examples, features):
    """Convert `human_pose_util.dataset.Example`s to `tf.train.Example`s."""
    feature_adder = {}
    for k, v in features.items():
        # FixedLenFeature
        # FixedLenSequenceFeature
        # VarLenFeature
        # SparseFeature
        if not isinstance(v, (tf.FixedLenFeature,
                              tf.FixedLenSequenceFeature,
                              tf.VarLenFeature)):
            raise NotImplementedError()
        is_list = len(v.shape) > 0 or isinstance(
                v, tf.FixedLenSequenceFeature)
        if v.dtype == tf.float32:
            if is_list:
                feature_adder[k] = lambda f, data: \
                    f.float32_list.extend(data)
            else:
                feature_adder[k] = lambda f, data: \
                    f.float32_list.append(data)
        elif v.dtype == tf.int64:
            if is_list:
                feature_adder[k] = lambda f, data: \
                    f.int64_list.extend(data)
            else:
                feature_adder[k] = lambda f, data: \
                    f.int64_list.append(data)
        elif v.dtype == tf.string:
            assert(is_list)
            feature_adder[k] = lambda f, data: \
                f.bytes_list.extend(data.tostring())
        else:
            raise NotImplementedError(
                'No FixedLenFeature for dtype %s' % str(v.dtype))
    for example in examples:
        tf_example = tf.train.Example()
        for k in features:
            feature_adder[k](tf_example.features[k], example[k])
            yield example


def dataset_to_tf_records(examples, features, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for example in examples_to_tf(examples, features):
            writer.write(example.SerializeToString())
