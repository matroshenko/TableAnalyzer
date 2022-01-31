import igraph

from datasets.ICDAR.rect import Rect


class CellsStructureBuilder(object):
    def __init__(self, merge_right_mask, merge_down_mask):
        self._merge_right_mask = merge_right_mask
        self._merge_down_mask = merge_down_mask
        self._rows_count = merge_right_mask.shape[0]
        self._cols_count = merge_down_mask.shape[1]

    def build(self):
        initial_cells = self._build_initial_cells()
        # TODO: Not implemented
        pass

    def _build_initial_cells(self):
        graph = self._create_graph()
        components = graph.components()
        return [self._get_component_rect(c) for c in components]
        
    def _create_graph(self):
        graph = igraph.Graph()
        graph.add_vertices(self._rows_count * self._cols_count)
        edges = []

        for i in range(self._rows_count):
            for j in range(self._cols_count-1):
                if self._merge_right_mask[i][j]:
                    edges.append((
                        self._to_1d_index(i, j), 
                        self._to_1d_index(i, j+1)
                    ))

        for i in range(self._rows_count-1):
            for j in range(self._cols_count):
                if self._merge_down_mask[i][j]:
                    edges.append((
                        self._to_1d_index(i, j), 
                        self._to_1d_index(i+1, j)
                    ))
        graph.add_edges(edges)

        return graph

    def _to_2d_index(self, index):
        assert 0 <= index and index < self._rows_count * self._cols_count
        return divmod(index, self._cols_count)

    def _to_1d_index(self, i, j):
        assert 0 <= i and i < self._rows_count
        assert 0 <= j and j < self._cols_count
        return i * self._cols_count + j

    def _get_component_rect(self, component):
        assert component
        left = min(self._to_2d_index(v)[1] for v in component)
        top = min(self._to_2d_index(v)[0] for v in component)
        right = max(self._to_2d_index(v)[1] for v in component) + 1
        bottom = max(self._to_2d_index(v)[0] for v in component) + 1
        return Rect(left, top, right, bottom)