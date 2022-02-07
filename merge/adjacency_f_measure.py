from enum import Enum
from collections import Counter

import tensorflow.keras as keras


class Direction(Enum):
    Horizontal = 0
    Vertical = 1


class AjacencyRelation:
    def __init__(self, leftOrTopCell, rightOrBottomCell, direction):
        self.leftOrTopCell = leftOrTopCell
        self.rightOrBottomCell = rightOrBottomCell
        self.direction = direction


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
        markup_adj_relations_list = self._create_markup_adj_relations_list(markup_table)
        markup_adj_relations_count = len(markup_adj_relations_list)

        table_cell_intersections_counter = self._create_table_cell_intersections_counter(
            markup_table, detected_grid, detected_cells)
        markup_text_rect_to_table_cell_dict = self._create_markup_text_rect_to_table_cell_dict(
            markup_table, detected_grid, detected_cells, table_cell_intersections_counter)

        correct_adj_relations_count = self._calculate_correct_adj_relations_count(
            markup_adj_relations_list, markup_text_rect_to_table_cell_dict)

        detected_adj_relations_count = self._calculate_detected_adj_relations_count(
            detected_cells, table_cell_intersections_counter)

        assert correct_adj_relations_count <= markup_adj_relations_count
        assert correct_adj_relations_count <= detected_adj_relations_count
        assert markup_adj_relations_count > 0

        self._markup_adj_relations_count.assign_add(markup_adj_relations_count)
        self._correct_adj_relations_count.assign_add(correct_adj_relations_count)
        self._detected_adj_relations_count.assign_add(detected_adj_relations_count)

    def result(self):
        if self._correct_adj_relations_count == 0:
            return 0
        
        assert self._markup_adj_relations_count > 0
        assert self._detected_adj_relations_count > 0

        recall = self._correct_adj_relations_count / self._markup_adj_relations_count
        precision = self._correct_adj_relations_count / self._detected_adj_relations_count
        return 2 * recall * precision / (recall + precision)

    def _create_markup_adj_relations_list(self, markup_table):
        result = []
        cells = markup_table.cells
        for i in range(len(cells) - 1):
            for j in range(i + 1, len(cells)):
                are_adjacent, direction = self._are_adjacent(
                    cells[i].grid_rect, cells[j].grid_rect)
                if are_adjacent:
                    result.append(AjacencyRelation(cells[i], cells[j], direction))

        return result

    def _are_adjacent(self, leftOrTopCell, rightOrBottomCell):
        if leftOrTopCell.right == rightOrBottomCell.left and leftOrTopCell.overlaps_vertically(rightOrBottomCell):
            return True, Direction.Horizontal
        if leftOrTopCell.bottom == rightOrBottomCell.top and leftOrTopCell.overlaps_horizontally(rightOrBottomCell):
            return True, Direction.Vertical
        return False, None

    def _create_table_cell_intersections_counter(self, markup_table, detected_grid, detected_cells):
        result = Counter()
        for cell in markup_table.cells:
            text_rect = cell.text_rect
            for detected_cell in detected_cells:
                detected_cell_rect = detected_grid.get_cell_rect(detected_cell)
                if text_rect.intersects(detected_cell_rect):
                    result[detected_cell] += 1

        return result

    def _create_markup_text_rect_to_table_cell_dict(
            self, markup_table, detected_grid, detected_cells, table_cell_intersections_counter):
        result = dict()
        for cell in markup_table.cells:
            text_rect = cell.text_rect
            for detected_cell in detected_cells:
                detected_cell_rect = detected_grid.get_cell_rect(detected_cell)
                if table_cell_intersections_counter[detected_cell] != 1:
                    continue
                if detected_cell_rect.contains(text_rect):
                    result[text_rect] = detected_cell

        return result

    def _calculate_correct_adj_relations_count(
            self, markup_adj_relations_list, markup_text_rect_to_table_cell_dict):
        
        result = 0
        for relation in markup_adj_relations_list:
            leftOrTopCellTextRect = relation.leftOrTopCell.text_rect
            rightOrBottomCellTextRect = relation.rightOrBottomCell.text_rect
            if leftOrTopCellTextRect not in markup_text_rect_to_table_cell_dict:
                continue
            if rightOrBottomCellTextRect not in markup_text_rect_to_table_cell_dict:
                continue
            leftOrTopTableCell = markup_text_rect_to_table_cell_dict[leftOrTopCellTextRect]
            rightOrBottomTableCell = markup_text_rect_to_table_cell_dict[rightOrBottomCellTextRect]
            are_adjacent, direction = self._are_adjacent(leftOrTopTableCell, rightOrBottomTableCell)
            if are_adjacent and direction == relation.direction:
                result += 1

        return result

    def _calculate_detected_adj_relations_count(self, detected_cells, table_cell_intersections_counter):
        result = 0
        for i in range(len(detected_cells) - 1):
            first_cell = detected_cells[i]
            if first_cell not in table_cell_intersections_counter:
                continue

            for j in range(i + 1, len(detected_cells)):
                second_cell = detected_cells[j]
                if second_cell not in table_cell_intersections_counter:
                    continue

                are_adjacent, _ = self._are_adjacent(
                    first_cell, second_cell)
                if are_adjacent:
                    result += 1

        return result