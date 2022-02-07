from unittest import TestCase, main
import time
import tensorflow as tf

import context
from merge.adjacency_f_measure import AdjacencyFMeasure
from datasets.ICDAR.markup_table import Cell, Table
from table.grid_structure import GridStructure
from utils.rect import Rect

class AdjacencyFMeasureTestCase(TestCase):
    def test_split_cell(self):
        markup_cells = [
            Cell(Rect(1, 1, 3, 2), Rect(0, 0, 1, 1)), 
            Cell(Rect(5, 1, 11, 2), Rect(1, 0, 3, 1)),

            Cell(Rect(1, 4, 3, 5), Rect(0, 1, 1, 2)),
            Cell(Rect(5, 4, 7, 5), Rect(1, 1, 2, 2)),
            Cell(Rect(9, 4, 11, 5), Rect(2, 1, 3, 2)),

            Cell(Rect(1, 7, 3, 8), Rect(0, 2, 1, 3)),
            Cell(Rect(5, 7, 7, 8), Rect(1, 2, 2, 3)),
            Cell(Rect(9, 7, 11, 8), Rect(2, 2, 3, 3))
        ]
        markup_table = Table(0, Rect(0, 0, 12, 9), markup_cells)

        detected_grid = GridStructure([0, 3, 6, 9], [0, 4, 8, 12])
        detected_cells = [
            Rect(0, 0, 1, 1), Rect(1, 0, 2, 1), Rect(2, 0, 3, 1),
            Rect(0, 1, 1, 2), Rect(1, 1, 2, 2), Rect(2, 1, 3, 2),
            Rect(0, 2, 1, 3), Rect(1, 2, 2, 3), Rect(2, 2, 3, 3)
        ]

        metric = AdjacencyFMeasure()
        metric.update_state(markup_table, detected_grid, detected_cells)

        # Recall = 8/11
        # Precision = 8/12
        self.assertEqual(metric.result(), 16/23)

    def test_split_column(self):
        markup_cells = [
            Cell(Rect(1, 1, 3, 2), Rect(0, 0, 1, 1)),
            Cell(Rect(4, 3, 5, 4), Rect(0, 1, 1, 2)),
            Cell(Rect(4, 5, 5, 6), Rect(0, 2, 1, 3))
        ]
        markup_table = Table(0, Rect(0, 0, 5, 7), markup_cells)

        detected_grid = GridStructure([0, 2, 4, 7], [0, 4, 6])
        detected_cells = [
            Rect(0, 0, 1, 1), Rect(1, 0, 2, 1),
            Rect(0, 1, 1, 2), Rect(1, 1, 2, 2),
            Rect(0, 2, 1, 3), Rect(1, 2, 2, 3)
        ]

        metric = AdjacencyFMeasure()
        metric.update_state(markup_table, detected_grid, detected_cells)

        # Recall = 1/2
        # Precision = 1
        self.assertEqual(metric.result(), 2/3)

    def test_shifted_split_point(self):
        markup_cells = [
            Cell(Rect(1, 1, 3, 2), Rect(0, 0, 1, 1)),
            Cell(Rect(5, 1, 7, 2), Rect(1, 0, 2, 1))
        ]
        markup_table = Table(0, Rect(0, 0, 8, 3), markup_cells)

        detected_grid = GridStructure([0, 3], [0, 6, 8])
        detected_cells = [Rect(0, 0, 1, 1), Rect(1, 0, 2, 1)]

        metric = AdjacencyFMeasure()
        metric.update_state(markup_table, detected_grid, detected_cells)

        # Recall = 0/1
        # Precision = 0/1
        self.assertEqual(metric.result(), 0)

if __name__ == '__main__':
    main()