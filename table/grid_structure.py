from utils.rect import Rect
from utils.interval import get_intervals_of_ones


class GridStructure(object):
    def __init__(self, h_positions, v_positions):
        assert len(h_positions) >= 2
        assert len(v_positions) >= 2
        assert all(h_positions[i] <= h_positions[i+1] for i in range(len(h_positions) - 1))
        assert all(v_positions[i] <= v_positions[i+1] for i in range(len(v_positions) - 1))

        self._h_positions = h_positions
        self._v_positions = v_positions

    def get_rows_count(self):
        return len(self._h_positions) - 1

    def get_cols_count(self):
        return len(self._v_positions) - 1    

    def get_cell_rect(self, cell):
        """Convert cell rect from grid coordinates to real coordinates."""
        assert 0 <= cell.top and cell.bottom <= self.get_rows_count()
        assert 0 <= cell.left and cell.right <= self.get_cols_count()
        return Rect(
            self._v_positions[cell.left],
            self._h_positions[cell.top],
            self._v_positions[cell.right],
            self._h_positions[cell.bottom]
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


class GridStructureBuilder(object):
    def __init__(self, rect, h_mask, v_mask):
        assert rect.get_height() == len(h_mask)
        assert rect.get_width() == len(v_mask)

        self._rect = rect
        self._h_mask = h_mask
        self._v_mask = v_mask

    def build(self):
        h_intervals = get_intervals_of_ones(self._h_mask)
        v_intervals = get_intervals_of_ones(self._v_mask)
        rect = self._rect
        return GridStructure(
            [rect.top] + [rect.top + interval.get_center() for interval in h_intervals] + [rect.bottom],
            [rect.left] + [rect.left + interval.get_center() for interval in v_intervals] + [rect.right]
        )