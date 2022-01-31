from unittest import TestCase, main

import context
from datasets.ICDAR.grid_structure import GridStructure, GridStructureBuilder
from datasets.ICDAR.rect import Rect


class GridStructureTest(TestCase):
    def test_create_by_rect_and_masks(self):
        rect = Rect(2, 1, 19, 12)
        h_mask = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
        v_mask = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
        grid = GridStructureBuilder(rect, h_mask, v_mask).build()
        expected_grid = GridStructure([1, 3, 5, 7, 9, 12], [2, 7, 11, 15, 19])
        self.assertEqual(grid, expected_grid)


if __name__ == '__main__':
    main()