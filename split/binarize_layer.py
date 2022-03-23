import tensorflow as tf
from tensorflow import keras

ops_module = tf.load_op_library('ops/ops.so')


class BinarizeLayer(keras.layers.Layer):
    """Binarize input probabilities via graph-cut algorithm."""
    def __init__(self, gc_lambda, name=None):
        super().__init__(trainable=False, name=name)
        assert gc_lambda >= 0
        self.gc_lambda = gc_lambda

    def call(self, probs):
        result = ops_module.gc_binarize(probs[0], self.gc_lambda)
        return tf.expand_dims(result, axis=0)
