from unittest import TestCase, main

import numpy as np

import context
from datasets.ICDAR.cells_structure import CellsStructureBuilder
from datasets.ICDAR.rect import Rect

class TestCellsStructure(TestCase):
    def test_simple(self):
        merge_right_mask = np.array([
            [True, True, False, False],
            [True, False, True, False],
            [False, False, False, False]
        ])
        merge_down_mask = np.array([
            [True, True, False, False, False],
            [False, False, False, False, False]
        ])
        expected_cells = [
            Rect(0, 0, 4, 2), 
            Rect(4, 0, 5, 1), 
            Rect(4, 1, 5, 2),
            Rect(0, 2, 1, 3),
            Rect(1, 2, 2, 3),
            Rect(2, 2, 3, 3),
            Rect(3, 2, 4, 3),
            Rect(4, 2, 5, 3)
        ]
        cells = CellsStructureBuilder(merge_right_mask, merge_down_mask).build()
        self.assertListEqual(sorted(expected_cells), sorted(cells))


if __name__ == '__main__':
    main()