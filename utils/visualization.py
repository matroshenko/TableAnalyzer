from PIL import Image, ImageDraw

from utils.rect import Rect
from table.grid_structure import GridStructure
from utils.interval import get_intervals_of_ones


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
        table_image, h_positions, v_positions, cells_grid_rects):
    width, height = table_image.size
    grid_structure = GridStructure(
        [0] + list(h_positions) + [height],
        [0] + list(v_positions) + [width])
    
    result_image = table_image.copy()
    draw = ImageDraw.Draw(result_image)
    for cell in cells_grid_rects:
        left, top, right, bottom = cell
        rect = grid_structure.get_cell_rect(Rect(left, top, right, bottom))
        draw.rectangle(rect.as_tuple(), outline=(255, 0, 0))
    return result_image

def create_markup_text_image(table_image, markup_table):
    markup_text_image = table_image.copy()
    draw = ImageDraw.Draw(markup_text_image)

    left = markup_table.rect.left
    top = markup_table.rect.top
    for cell in markup_table.cells:
        text_rect = cell.text_rect
        shifted_text_rect = Rect(
            text_rect.left - left,
            text_rect.top - top,
            text_rect.right - left,
            text_rect.bottom - top
        )
        draw.rectangle(shifted_text_rect.as_tuple(), fill=(255, 0, 0), outline=(0, 255, 0))

    return Image.blend(table_image, markup_text_image, 0.5)