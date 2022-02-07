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

  def intersects(self, other):
    return self.overlaps_horizontally(other) and self.overlaps_vertically(other)

  def overlaps_horizontally(self, other):
    return self.left < other.right and other.left < self.right
  
  def overlaps_vertically(self, other):
    return self.top < other.bottom and other.top < self.bottom

  def __eq__(self, other):
    return self.as_tuple() == other.as_tuple()

  def __lt__(self, other):
    return self.as_tuple() < other.as_tuple()

  def __or__(self, other):
    left = min(self.left, other.left)
    top = min(self.top, other.top)
    right = max(self.right, other.right)
    bottom = max(self.bottom, other.bottom)
    return Rect(left, top, right, bottom)

  def __repr__(self):
    return '[{}, {}, {}, {}]'.format(self.left, self.top, self.right, self.bottom)

  def __hash__(self):
    return hash(self.as_tuple())
