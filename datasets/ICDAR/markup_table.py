import pickle

import tensorflow as tf
import numpy as np

from utils import Interval
from rect import Rect


class Cell(object):
  def __init__(self, text_rect, grid_rect):
    self.text_rect = text_rect
    self.grid_rect = grid_rect

  def __eq__(self, other):
    return (
      self.text_rect == other.text_rect
      and self.grid_rect == other.grid_rect
    )


class Table(object):
  def __init__(self, id, rect, cells):
    self.id = id
    self.rect = rect
    self.cells = cells

  def __eq__(self, other):
    return (
      self.id == other.id
      and self.rect == other.rect
      and self.cells == other.cells
    )

  def to_tensor(self):
    bytes = pickle.dumps(self)
    return tf.constant(bytes)

  @staticmethod
  def from_tensor(tensor):
    bytes = tensor.numpy()
    return pickle.loads(bytes)

  def create_horz_split_points_mask(self):
    height = self.rect.bottom - self.rect.top
    shift = self.rect.top
    result = np.zeros(shape=(height,), dtype=np.bool)

    split_point_indexes = self._get_horz_split_points_indexes()
    assert len(split_point_indexes) >= 2
    # Iterate only internal split point indexes.
    for i in range(1, len(split_point_indexes)-1):
      split_point_index = split_point_indexes[i]
      interval = self._get_horz_split_point_interval(split_point_index)
      result[interval.start - shift : interval.end - shift] = True

    return result

  def create_vert_split_points_mask(self):
    width = self.rect.right - self.rect.left
    shift = self.rect.left
    result = np.zeros(shape=(width,), dtype=np.bool)

    split_point_indexes = self._get_vert_split_points_indexes()
    assert len(split_point_indexes) >= 2
    # Iterate only internal split point indexes.
    for i in range(1, len(split_point_indexes)-1):
      split_point_index = split_point_indexes[i]
      interval = self._get_vert_split_point_interval(split_point_index)
      result[interval.start - shift : interval.end - shift] = True

    return result

  def create_merge_masks(self, grid):
    assert self.rect == grid.get_bounding_rect()

    n = grid.get_rows_count()
    m = grid.get_cols_count()
    merge_right_mask = np.zeros(shape=(n, m-1), dtype=np.bool)
    merge_down_mask = np.zeros(shape=(n-1, m), dtype=np.bool)

    for cell in self.cells:
      outer_rect = self._calculate_outer_rect(cell)
      for i in range(n):
        for j in range(m):
          cell = Rect(j, i, j+1, i+1)
          if not outer_rect.contains(grid.get_cell_rect(cell)):
            continue
          
          right_cell = Rect(j+1, i, j+2, i+1)
          if j+1 < m and outer_rect.contains(grid.get_cell_rect(right_cell)):
            merge_right_mask[i][j] = True
            
          bottom_cell = Rect(j, i+1, j+1, i+2)
          if i+1 < n and outer_rect.contains(grid.get_cell_rect(bottom_cell)):
            merge_down_mask[i][j] = True

    return merge_right_mask, merge_down_mask

  def _get_horz_split_points_indexes(self):
    result = set()
    for cell in self.cells:
      result.add(cell.grid_rect.top)
      result.add(cell.grid_rect.bottom)
    return sorted(result)

  def _get_vert_split_points_indexes(self):
    result = set()
    for cell in self.cells:
      result.add(cell.grid_rect.left)
      result.add(cell.grid_rect.right)
    return sorted(result)

  def _get_left_adjacent_cells(self, vert_split_point_index):
    result = []
    for cell in self.cells:
      if cell.grid_rect.right == vert_split_point_index:
        result.append(cell)
    return result

  def _get_right_adjacent_cells(self, vert_split_point_index):
    result = []
    for cell in self.cells:
      if cell.grid_rect.left == vert_split_point_index:
        result.append(cell)
    return result

  def _get_top_adjacent_cells(self, horz_split_point_index):
    result = []
    for cell in self.cells:
      if cell.grid_rect.bottom == horz_split_point_index:
        result.append(cell)
    return result

  def _get_bottom_adjacent_cells(self, horz_split_point_index):
    result = []
    for cell in self.cells:
      if cell.grid_rect.top == horz_split_point_index:
        result.append(cell)
    return result

  def _get_horz_split_point_interval(self, split_point_index):
    top_adjacent_cells = self._get_top_adjacent_cells(split_point_index)
    bottom_adjacent_cells = self._get_bottom_adjacent_cells(split_point_index)

    # Adjacent row could be empty (empty cells are not stored).
    # In this case we suppose, that interval has length=1.
    assert top_adjacent_cells or bottom_adjacent_cells

    start = None
    end = None
    if top_adjacent_cells:
      start = max(cell.text_rect.bottom for cell in top_adjacent_cells)
    if bottom_adjacent_cells:
      end = min(cell.text_rect.top for cell in bottom_adjacent_cells)
    if start is None:
      assert end is not None
      start = end - 1
    if end is None:
      assert start is not None
      end = start + 1
    if end <= start:
      # In this case we suppose, that interval has length=1.
      end = start+1

    return Interval(start, end)

  def _get_vert_split_point_interval(self, split_point_index):
    left_adjacent_cells = self._get_left_adjacent_cells(split_point_index)
    right_adjacent_cells = self._get_right_adjacent_cells(split_point_index)
    assert left_adjacent_cells
    assert right_adjacent_cells

    start = max(cell.text_rect.right for cell in left_adjacent_cells)
    end = min(cell.text_rect.left for cell in right_adjacent_cells)
    
    if end <= start:
      # In this case we suppose, that interval has length=1.
      end = start + 1

    return Interval(start, end)

  def _calculate_outer_rect(self, cell):
    table_rect = self.rect
    grid_rect = cell.grid_rect
    horz_split_points_indexes = self._get_horz_split_points_indexes()
    vert_split_points_indexes = self._get_vert_split_points_indexes()
    is_adjacent_to_left_border = grid_rect.left == vert_split_points_indexes[0]
    is_adjacent_to_top_border = grid_rect.top == horz_split_points_indexes[0]
    is_adjacent_to_right_border = grid_rect.right == vert_split_points_indexes[-1]
    is_adjacent_to_bottom_border = grid_rect.bottom == horz_split_points_indexes[-1]
    
    left = table_rect.left if is_adjacent_to_left_border else self._get_vert_split_point_interval(grid_rect.left).start
    top = table_rect.top if is_adjacent_to_top_border else self._get_horz_split_point_interval(grid_rect.top).start
    right = table_rect.right if is_adjacent_to_right_border else self._get_vert_split_point_interval(grid_rect.right).end
    bottom = table_rect.bottom if is_adjacent_to_bottom_border else self._get_horz_split_point_interval(grid_rect.bottom).end

    return Rect(left, top, right, bottom)