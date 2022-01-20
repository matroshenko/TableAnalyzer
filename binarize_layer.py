import tensorflow as tf
from tensorflow import keras
import networkx as nx

class BinarizeLayer(keras.layers.Layer):
    """Binarize input probabilities via graph-cut algorithm."""
    def __init__(self, gc_lambda=0.75):
        super().__init__(trainable=False)
        self.gc_lambda = gc_lambda

    def call(self, inputs):
        graph, source, sink = self._create_graph(inputs)
        cut_value, (reachable_nodes, non_reachable_nodes) = nx.minimum_cut(graph, source, sink)
        result = [0] * len(inputs)
        for node in reachable_nodes:
            result[node-1] = 1
        return tf.constant(result, dtype=tf.int32)

    def _create_graph(self, inputs):
        result = nx.Graph()
        source = 0
        sink = len(inputs) + 1
        for i, prob in enumerate(inputs):
            result.add_edge(source, i+1, capacity=prob)
            result.add_edge(i+1, sink, capacity=1-prob)
            if i > 0:
                result.add_edge(i, i+1, capacity=self.gc_lambda)
        return result, source, sink
