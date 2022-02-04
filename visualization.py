from PIL import Image, ImageDraw

from rect import Rect
from table.grid_structure import GridStructureBuilder
from table.cells_structure import CellsStructureBuilder
from utils import get_intervals_of_ones


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