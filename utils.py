from PIL import Image, ImageDraw

from datasets.ICDAR.rect import Rect
from datasets.ICDAR.grid_structure import GridStructureBuilder
from datasets.ICDAR.cells_structure import CellsStructureBuilder


class Interval(object):
    def __init__(self, start, end):
        assert start < end
        self.start = start
        self.end = end

    def get_length(self):
        return self.end - self.start

    def get_center(self):
        return (self.start + self.end) // 2

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

def create_split_result_image(table_image, horz_split_points_mask, vert_split_points_mask):
    height = len(horz_split_points_mask)
    width = len(vert_split_points_mask)
    assert table_image.size == (width, height)
    split_points_image = table_image.copy()
    draw = ImageDraw.Draw(split_points_image)

    for interval in get_intervals_of_ones(horz_split_points_mask):
        draw.rectangle((0, interval.start, width, interval.end), fill=(255, 0, 0))

    for interval in get_intervals_of_ones(vert_split_points_mask):
        draw.rectangle((interval.start, 0, interval.end, height), fill=(255, 0, 0))
    
    return Image.blend(table_image, split_points_image, 0.5)

def create_merge_result_image(
        table_image, horz_split_points_mask, vert_split_points_mask,
        merge_right_mask, merge_down_mask):
    height = len(horz_split_points_mask)
    width = len(vert_split_points_mask)
    assert table_image.size == (width, height)
    grid_structure = GridStructureBuilder(
        Rect(0, 0, width, height), horz_split_points_mask, vert_split_points_mask).build()
    cells_structure = CellsStructureBuilder(merge_right_mask, merge_down_mask).build()
    
    result_image = table_image.copy()
    draw = ImageDraw.Draw(result_image)
    for cell in cells_structure:
        rect = grid_structure.get_cell_rect(cell)
        draw.rectangle(rect.as_tuple(), outline=(255, 0, 0))
    return result_image