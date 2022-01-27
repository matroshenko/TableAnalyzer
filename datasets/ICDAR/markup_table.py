import numpy as np

class Rect(object):
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def as_tuple(self):
        return (self.left, self.top, self.right, self.bottom)


class Cell(object):
  def __init__(self, text_rect, grid_rect):
    self.text_rect = text_rect
    self.grid_rect = grid_rect


class Table(object):
  def __init__(self, id, rect, cells):
    self.id = id
    self.rect = rect
    self.cells = cells

  def create_horz_split_points_mask(self):
    height = self.rect.bottom - self.rect.top
    result = np.zeros(shape=(height,), dtype=np.bool)

    split_point_indexes = self._get_horz_split_points_indexes()
    assert len(split_point_indexes) >= 2
    # Iterate only internal split point indexes.
    for i in range(1, len(split_point_indexes)-1):
      split_point_index = split_point_indexes[i]
      top_adjacent_cells = self._get_top_adjacent_cells(split_point_index)
      bottom_adjacent_cells = self._get_bottom_adjacent_cells(split_point_index)
      assert top_adjacent_cells
      assert bottom_adjacent_cells
      split_point_interval = (
        max(cell.text_rect.bottom - self.rect.top for cell in top_adjacent_cells), 
        min(cell.text_rect.top - self.rect.top for cell in bottom_adjacent_cells) + 1
      )
      result[split_point_interval[0] : split_point_interval[1]] = True

    return result

  def create_vert_split_points_mask(self):
    width = self.rect.right - self.rect.left
    result = np.zeros(shape=(width,), dtype=np.bool)

    split_point_indexes = self._get_vert_split_points_indexes()
    assert len(split_point_indexes) >= 2
    # Iterate only internal split point indexes.
    for i in range(1, len(split_point_indexes)-1):
      split_point_index = split_point_indexes[i]
      left_adjacent_cells = self._get_left_adjacent_cells(split_point_index)
      right_adjacent_cells = self._get_right_adjacent_cells(split_point_index)
      assert left_adjacent_cells
      assert right_adjacent_cells
      split_point_interval = (
        max(cell.text_rect.right - self.rect.left for cell in left_adjacent_cells), 
        min(cell.text_rect.left - self.rect.left for cell in right_adjacent_cells) + 1
      )
      result[split_point_interval[0] : split_point_interval[1]] = True

    return result

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
    if result:
      return result

    # For lower explicit horz split point of the double split point
    # there will be no adjacent cells.
    for cell in self.cells:
      if cell.grid_rect.bottom == horz_split_point_index - 1:
        result.append(cell)
    return result

  def _get_bottom_adjacent_cells(self, horz_split_point_index):
    result = []
    for cell in self.cells:
      if cell.grid_rect.top == horz_split_point_index:
        result.append(cell)
    if result:
      return result

    # For upper explicit horz split point of the double split point
    # there will be no adjacent cells.
    for cell in self.cells:
      if cell.grid_rect.top == horz_split_point_index + 1:
        result.append(cell)
    return result

