from unittest import TestCase, main

import numpy as np

import context
from datasets.ICDAR.markup_table import Rect, Cell, Table


class TableTest(TestCase):
    def setUp(self):
        # Creating simple table with 4 columns and 5 rows.
        rect = Rect(2, 1, 19, 12)
        cells = [
            Cell(Rect(3, 2, 6, 4), Rect(0, 0, 1, 2)),
            Cell(Rect(8, 2, 18, 3), Rect(1, 0, 4, 1)),
            Cell(Rect(8, 4, 10, 5), Rect(1, 1, 2, 2)),
            Cell(Rect(12, 4, 14, 5), Rect(2, 1, 3, 2)),
            Cell(Rect(16, 4, 18, 5), Rect(3, 1, 4, 2)),

            Cell(Rect(3, 6, 5, 7), Rect(0, 2, 1, 3)),
            Cell(Rect(9, 6, 10, 7), Rect(1, 2, 2, 3)),
            Cell(Rect(13, 6, 14, 7), Rect(2, 2, 3, 3)),
            Cell(Rect(17, 6, 18, 7), Rect(3, 2, 4, 3)),

            Cell(Rect(3, 8, 5, 9), Rect(0, 3, 1, 4)),
            Cell(Rect(9, 8, 10, 9), Rect(1, 3, 2, 4)),
            Cell(Rect(13, 8, 14, 9), Rect(2, 3, 3, 4)),
            Cell(Rect(17, 8, 18, 9), Rect(3, 3, 4, 4)),

            Cell(Rect(3, 10, 5, 11), Rect(0, 4, 1, 5)),
            Cell(Rect(9, 10, 10, 11), Rect(1, 4, 2, 5)),
            Cell(Rect(13, 10, 14, 11), Rect(2, 4, 3, 5)),
            Cell(Rect(17, 10, 18, 11), Rect(3, 4, 4, 5))
        ]
        self._table = Table(0, rect, cells)

    def test_horz_split_points_mask(self):
        mask = self._table.create_horz_split_points_mask()
        expected_mask = np.array(
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.bool)
        self.assertTrue(np.all(mask == expected_mask))

    def test_vert_split_points_mask(self):
        mask = self._table.create_vert_split_points_mask()
        expected_mask = np.array(
            [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.bool)
        self.assertTrue(np.all(mask == expected_mask))


if __name__ == '__main__':
    main()