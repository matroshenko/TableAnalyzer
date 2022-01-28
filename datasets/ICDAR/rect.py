class Rect(object):
  def __init__(self, left, top, right, bottom):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom

  def get_width(self):
    return self.right - self.left

  def get_height(self):
    return self.bottom - self.top

  def as_tuple(self):
    return (self.left, self.top, self.right, self.bottom)

  def contains(self, other):
    return (
      self.left <= other.left and other.right <= self.right
      and self.top <= other.top and other.bottom <= self.bottom
    )

  def __eq__(self, other):
    return self.as_tuple() == other.as_tuple()