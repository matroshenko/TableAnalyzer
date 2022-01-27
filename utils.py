import PIL


class Interval(object):
    def __init__(self, start, end):
        assert start < end
        self.start = start
        self.end = end

    def get_length(self):
        return self.end - self.start

    @staticmethod
    def get_intersection_length(first, second):
        return max(0, min(first.end, second.end) - max(first.start, second.start))

  
def get_intervals_of_ones(mask):
    result = []
    current_inteval_start = None
    is_inside_interval = False
    for i in range(len(mask)):
        if mask[i] == 1:
            if not is_inside_interval:
                current_inteval_start = i
                is_inside_interval = True
        else:
            if is_inside_interval:
                assert current_inteval_start is not None
                result.append(Interval(current_inteval_start, i))
                is_inside_interval = False
    if is_inside_interval:
        assert current_inteval_start is not None
        result.append(Interval(current_inteval_start, len(mask)))
    return result


def create_debug_image(table_image, horz_split_points_mask, vert_split_points_mask):
    height = len(horz_split_points_mask)
    width = len(vert_split_points_mask)
    split_points_image = PIL.Image.new('RGB', (width, height))
    pixels = split_points_image.load()
    for x in range(width):
      for y in range(height):
        if horz_split_points_mask[y] or vert_split_points_mask[x]:
          pixels[x, y] = (255, 0, 0)
    return PIL.Image.blend(table_image, split_points_image, 0.5)