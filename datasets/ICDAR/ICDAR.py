"""ICDAR dataset."""

from collections import namedtuple
import tensorflow_datasets as tfds
import xml.etree.ElementTree as ET
import pdf2image
import PIL
import io

# TODO(ICDAR): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(ICDAR): BibTeX citation
_CITATION = """
"""

class Icdar(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ICDAR dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'horz_split_points': tfds.features.Image(shape=(None, None, 1)),
            'vert_split_points': tfds.features.Image(shape=(None, None, 1))
        }),
        homepage='https://www.tamirhassan.com/html/dataset.html',
        citation=_CITATION,
        disable_shuffling=True
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    path = dl_manager.download_and_extract(
      'https://www.tamirhassan.com/html/files/icdar2013-competition-dataset-with-gt.zip')

    return {
        'train': self._generate_examples(path)
    }

  def _generate_examples(self, path):
    """Yields examples."""

    for pdf_file_path in path.glob('*.pdf'):
      stem = pdf_file_path.stem
      region_file_path = pdf_file_path.with_name(stem + '-reg.xml')
      structure_file_path = pdf_file_path.with_name(stem + '-str.xml')

      pages = pdf2image.convert_from_path(pdf_file_path, dpi=68)
      for page_number, table in self._generate_tables(region_file_path, structure_file_path):
        key = '{}-{}'.format(stem, page_number)
        page = pages[page_number]
        page_height = page.height
        dx = -5
        dy = 25
        crop_rect = (
          table.rect.left + dx, page_height - table.rect.bottom + dy, 
          table.rect.right + dx, page_height - table.rect.top + dy
          )
        table_image = page.crop(crop_rect)
        #table_image = page.crop(table.rect)
        page.save('page.png')
        table_image.save('table.png')
        yield key, {
          'image': self._image_to_byte_array(table_image),
          'horz_split_points': self._image_to_byte_array(table.create_horz_split_points_image()),
          'vert_split_points': self._image_to_byte_array(table.create_vert_split_points_image())
        }

  def _generate_tables(self, region_file_path, structure_file_path):
    regions_tree = ET.parse(region_file_path)
    structures_tree = ET.parse(structure_file_path)
    for table_node, table_structure_node in zip(regions_tree.getroot(), structures_tree.getroot()):
      region_node = table_node.find('region')
      page_number = int(region_node.get('page')) - 1
      table_rect = self._get_bounding_box(region_node)
      cells_node = table_structure_node.find('region')
      cells = [self._get_cell(node) for node in cells_node]
      yield page_number, Table(table_rect, cells)

  def _get_bounding_box(self, xml_node):
    bounding_box_node = xml_node.find('bounding-box')
    return Rect(
      int(bounding_box_node.get('x1')),
      int(bounding_box_node.get('y1')),
      int(bounding_box_node.get('x2')),
      int(bounding_box_node.get('y2'))
    )

  def _get_cell(self, xml_node):
    rect = self._get_bounding_box(xml_node)
    col_start = int(xml_node.get('start-col'))
    col_end = int(xml_node.get('end-col', col_start))
    row_start = int(xml_node.get('start-row'))
    row_end = int(xml_node.get('end-row', row_start))
    return Cell(rect, col_start, col_end, row_start, row_end)

  def _image_to_byte_array(self, image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


Rect = namedtuple('Rect', ['left', 'top', 'right', 'bottom'])


class Cell(object):
  def __init__(self, rect, col_start, col_end, row_start, row_end):
    self.rect = rect
    self.col_start = col_start
    self.col_end = col_end
    self.row_start = row_start
    self.row_end = row_end


class Table(object):
  def __init__(self, rect, cells):
    self.rect = rect
    self.cells = cells

  def create_horz_split_points_image(self):
    # TODO
    width = self.rect.right - self.rect.left
    height = self.rect.bottom - self.rect.top
    return PIL.Image.new('L', (width, height))

  def create_vert_split_points_image(self):
    # TODO
    width = self.rect.right - self.rect.left
    height = self.rect.bottom - self.rect.top
    return PIL.Image.new('L', (width, height))

