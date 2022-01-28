from datasets.ICDAR.rect import Rect


class Grid(object):
    def __init__(self, h_positions, v_positions):
        assert len(h_positions) >= 2
        assert len(v_positions) >= 2
        assert all(h_positions[i] < h_positions[i+1] for i in range(len(h_positions) - 1))
        assert all(v_positions[i] < v_positions[i+1] for i in range(len(v_positions) - 1))

        self._h_positions = h_positions
        self._v_positions = v_positions

    def get_rows_count(self):
        return len(self._h_positions) - 1

    def get_cols_count(self):
        return len(self._v_positions) - 1    

    def get_cell_rect(self, i, j):
        assert 0 <= i and i < self.get_rows_count()
        assert 0 <= j and j < self.get_cols_count()
        return Rect(
            self._v_positions[j],
            self._h_positions[i],
            self._v_positions[j+1],
            self._h_positions[i+1]
        )

    def get_bounding_rect(self):
        return Rect(
            self._v_positions[0],
            self._h_positions[0],
            self._v_positions[-1],
            self._h_positions[-1]
        )