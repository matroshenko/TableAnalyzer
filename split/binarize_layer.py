import tensorflow as tf
from tensorflow import keras
import numpy as np
import igraph

def get_capacity(value):
    return int(1024 * value)

class BinarizeLayer(keras.layers.Layer):
    """Binarize input probabilities via graph-cut algorithm."""
    def __init__(self, gc_lambda=0.75):
        super().__init__(trainable=False)
        assert gc_lambda >= 0
        self.gc_lambda = gc_lambda

    def call(self, probs):
        probs = probs.numpy()
        graph, source, sink = self._create_graph(probs)
        reachable_nodes, _ = graph.st_mincut(source, sink, 'capacity')
        reachable_nodes.remove(source)
        result = [0] * len(probs)
        for node in reachable_nodes:
            result[node-1] = 1
        return tf.constant(result, dtype=tf.int32)

    def _create_graph(self, probs):
        n = len(probs)
        assert n > 0
        result = igraph.Graph()
        source = 0
        sink = n + 1
        result.add_vertices(n + 2)

        result.add_edges( 
            [(source, i+1) for i in range(n)], 
            {'capacity': [get_capacity(p) for p in probs]} 
        )
        result.add_edges( 
            [(i+1, sink) for i in range(n)], 
            {'capacity': [get_capacity(1-p) for p in probs]} 
        )
        if n > 1:
            result.add_edges(
                [(i+1, i+2) for i in range(n-1)],
                {'capacity': [get_capacity(self.gc_lambda)] * (n-1)}
            )
        return result, source, sink
