import tensorflow as tf
from tensorflow import keras
import igraph

def get_integer_capacity(value):
    return int(1024*value)

class BinarizeLayer(keras.layers.Layer):
    """Binarize input probabilities via graph-cut algorithm."""
    def __init__(self, gc_lambda=0.75):
        super().__init__(trainable=False)
        assert gc_lambda >= 0
        self.gc_lambda = gc_lambda

    def call(self, inputs):
        graph, source, sink = self._create_graph(inputs)
        reachable_nodes, _ = graph.st_mincut(source, sink, 'capacity')
        reachable_nodes.remove(source)
        result = [0] * len(inputs)
        for node in reachable_nodes:
            result[node-1] = 1
        return tf.constant(result, dtype=tf.int32)

    def _create_graph(self, inputs):
        result = igraph.Graph()
        source = 0
        sink = len(inputs) + 1
        result.add_vertices(len(inputs) + 2)

        for i, prob in enumerate(inputs):
            result.add_edges([(source, i+1)], {'capacity': [get_integer_capacity(prob)]})
            result.add_edges([(i+1, sink)], {'capacity': [get_integer_capacity(1-prob)]})
            if i > 0:
                result.add_edges([(i, i+1)], {'capacity': [get_integer_capacity(self.gc_lambda)]})
        return result, source, sink
