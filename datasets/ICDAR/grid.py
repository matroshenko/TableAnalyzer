from datasets.ICDAR.rect import Rect
from utils import get_intervals_of_ones


class Grid(object):
    def __init__(self, h_positions, v_positions):
        assert len(h_positions) >= 2
        assert len(v_positions) >= 2
        assert all(h_positions[i] < h_positions[i+1] for i in range(len(h_positions) - 1))
        assert all(v_positions[i] < v_positions[i+1] for i in range(len(v_positions) - 1))

        self._h_positions = h_positions
        self._v_positions = v_positions

    @classmethod
    def create_by_rect_and_masks(cls, rect, h_mask, v_mask):
        assert rect.get_height() == len(h_mask)
        assert rect.get_width() == len(v_mask)
        h_intervals = get_intervals_of_ones(h_mask)
        v_intervals = get_intervals_of_ones(v_mask)
        return cls(
            [rect.top] + [rect.top + interval.get_center() for interval in h_intervals] + [rect.bottom],
            [rect.left] + [rect.left + interval.get_center() for interval in v_intervals] + [rect.right]
        )

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

    def __eq__(self, other):
        return (
            self._h_positions == other._h_positions 
            and self._v_positions == other._v_positions
        )