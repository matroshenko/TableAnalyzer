import PIL


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