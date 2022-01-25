import tensorflow.keras as keras
import networkx as nx
from networkx.algorithms import bipartite

class Interval(object):
    def __init__(self, start, end):
        assert start < end
        self.start = start
        self.end = end

    def get_length(self):
        return self.end - self.start

    @staticmethod
    def get_intersection_length(first, second):
        return max(0, min(first.end, second.end) - max(first.start, second.start))


class IntervalwiseFMeasure(keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.markup_intervals_count = self.add_weight('markup_intervals_count', initializer='zeros', dtype='int32')
        self.predicted_intervals_count = self.add_weight('predicted_intervals_count', initializer='zeros', dtype='int32')
        self.matched_intervals_count = self.add_weight('matched_intervals_count', initializer='zeros', dtype='int32')

    def update_state(self, markup_mask, predicted_mask, sample_weight=None):
        assert len(markup_mask.shape) == 2
        assert len(predicted_mask.shape) == 2
        for markup_mask_element, predicted_mask_element in zip(markup_mask, predicted_mask):
            markup_intervals = self._get_intervals_of_ones(markup_mask_element)
            predicted_intervals = self._get_intervals_of_ones(predicted_mask_element)
            matching_size = self._calculate_matching_size(markup_intervals, predicted_intervals)

            self.markup_intervals_count.assign_add(len(markup_intervals))
            self.predicted_intervals_count.assign_add(len(predicted_intervals))
            self.matched_intervals_count.assign_add(matching_size)

    def result(self):
        if self.matched_intervals_count == 0:
            return 0
        assert self.markup_intervals_count > 0
        assert self.predicted_intervals_count > 0
        recall = self.matched_intervals_count / self.markup_intervals_count
        precision = self.matched_intervals_count / self.predicted_intervals_count
        return 2 * recall * precision / (recall + precision)

    @staticmethod
    def _get_intervals_of_ones(mask):
        result = []
        current_inteval_start = None
        is_inside_interval = False
        for i in range(len(mask)):
            if mask[i].numpy() == 1:
                if not is_inside_interval:
                    current_inteval_start = i
                    is_inside_interval = True
            else:
                if is_inside_interval:
                    assert current_inteval_start is not None
                    result.append(Interval(current_inteval_start, i))
                    is_inside_interval = False
        if is_inside_interval:
            assert current_inteval_start is not None
            result.append(Interval(current_inteval_start, len(mask)))
        return result

    @staticmethod
    def _calculate_matching_size(first_intervals_list, second_intervals_list):
        graph = nx.Graph()
        n = len(first_intervals_list)
        m = len(second_intervals_list)
        for i in range(n):
            graph.add_node(i, bipartite=0)
        for i in range(m):
            graph.add_node(n+i, bipartite=1)
        for i, interval1 in enumerate(first_intervals_list):
            for j, interval2 in enumerate(second_intervals_list):
                intersection_length = Interval.get_intersection_length(interval1, interval2)
                min_length = min(interval1.get_length(), interval2.get_length())
                assert min_length > 0
                if intersection_length / min_length > 0.5:
                    graph.add_edge(i, n+j)
        matching_dict = bipartite.maximum_matching(graph, top_nodes=list(range(n)))
        assert len(matching_dict) % 2 == 0
        return len(matching_dict) // 2
