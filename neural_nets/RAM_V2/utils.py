import tensorflow.compat.v1 as tf


def sample_normal_single(mean, stddev, name=None):
    return tf.random_normal(
        # shape=mean.get_shape(),
        shape=tf.shape(mean),
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
        seed=None,
        name=name,
    )


def get_shape2D(in_val):
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))
