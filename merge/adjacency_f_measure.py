import tensorflow.keras as keras


class AdjacencyFMeasure(keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self._markup_adj_relations_count = self.add_weight(
            'markup_adj_relations_count', initializer='zeros', dtype='int32')
        self._correct_adj_relations_count = self.add_weight(
            'correct_adj_relations_count', initializer='zeros', dtype='int32')
        self._detected_adj_relations_count = self.add_weight(
            'detected_adj_relations_count', initializer='zeros', dtype='int32')

    def update_state(self, markup_table, detected_grid, detected_cells):
        pass

    def result(self):
        assert self._correct_adj_relations_count <= self._markup_adj_relations_count
        assert self._correct_adj_relations_count <= self._detected_adj_relations_count
        assert self._markup_adj_relations_count > 0

        if self._detected_adj_relations_count == 0:
            return 0

        recall = self._correct_adj_relations_count / self._markup_adj_relations_count
        precision = self._correct_adj_relations_count / self._detected_adj_relations_count
        return 2 * recall * precision / (recall + precision)