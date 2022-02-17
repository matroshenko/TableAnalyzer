"""ICDAR 2013 table recognition dataset."""

from abc import abstractmethod
import xml.etree.ElementTree as ET
import io
import os
import glob
import pathlib
from itertools import chain

import tensorflow_datasets as tfds
import tensorflow as tf
import pdf2image
import PIL

from table.markup_table import Cell, Table
from utils.rect import Rect
from table.grid_structure import GridStructureBuilder
import split.evaluation


# TODO(ICDAR): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(ICDAR): BibTeX citation
_CITATION = """
"""

_FILES_TO_IGNORE = [
  ['competition-dataset-eu', 'eu-015'],  # cells lie outside page rect
  ['competition-dataset-us', 'us-035a'],  # 2nd table has invalid cell coords
  ['eu-dataset', 'eu-032'], # 2nd table has invalid cell coords
  ['eu-dataset', 'eu-014'], # invalid cell text rect
  ['eu-dataset', 'eu-023'], # invalid cell text rect
  ['us-gov-dataset', 'us-025'], # invalid cell text rect
  ['us-gov-dataset', 'us-012'], # invalid cell text rect
  ['us-gov-dataset', 'us-020'], # invalid cell text rect
]


class IcdarBase(tfds.core.GeneratorBasedBuilder):
  """Base DatasetBuilder for ICDAR datasets."""

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=self._get_features_dict(),
        homepage='https://www.tamirhassan.com/html/dataset.html',
        citation=_CITATION,
        disable_shuffling=True
    )

  @abstractmethod
  def _get_features_dict(self) -> tfds.features.FeaturesDict:
    """Returns features, describing dataset element."""
    pass

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    pathes = dl_manager.download_and_extract(
      ['https://www.tamirhassan.com/html/files/eu-dataset-20130324.zip',
      'https://www.tamirhassan.com/html/files/us-gov-dataset-20130324.zip',
      'https://www.tamirhassan.com/html/files/icdar2013-competition-dataset-with-gt.zip'])

    if not isinstance(pathes, list):
      # During unit-testing dl_manager will return path to dummy_data.
      return {'train': self._generate_examples(pathes)}

    return {
        'train': chain(
          self._generate_examples(pathes[0]), 
          self._generate_examples(pathes[1])),
        'test': self._generate_examples(pathes[2])
    }

  def _generate_examples(self, path):
    """Yields examples."""

    for pdf_file_path in glob.glob(os.path.join(path, '**/*.pdf'), recursive=True):
      pdf_file_path = pathlib.Path(pdf_file_path)
      parent_folder_name = pdf_file_path.parts[-2]
      stem = pdf_file_path.stem
      if [parent_folder_name, stem] in _FILES_TO_IGNORE:
        continue
      
      region_file_path = pdf_file_path.with_name(stem + '-reg.xml')
      structure_file_path = pdf_file_path.with_name(stem + '-str.xml')

      pages = pdf2image.convert_from_path(pdf_file_path, dpi=72)
      for page_number, table in self._generate_tables(pages, region_file_path, structure_file_path):
        key = '{}-{}-{}'.format(parent_folder_name, stem, table.id)
        page = pages[page_number]
        table_image = page.crop(table.rect.as_tuple())
        yield key, self._get_single_example_dict(table_image, table)

  @abstractmethod
  def _get_single_example_dict(self, table_image, markup_table):
    """Returns dict with nessary inputs for the model."""
    pass

  def _generate_tables(self, pages, region_file_path, structure_file_path):
    regions_tree = ET.parse(region_file_path)
    structures_tree = ET.parse(structure_file_path)
    for table_node, table_structure_node in zip(regions_tree.getroot(), structures_tree.getroot()):
      table_id = int(table_node.get('id'))
      region_node = table_node.find('region')
      page_number = int(region_node.get('page')) - 1
      page_width, page_height = pages[page_number].size
      table_rect = self._get_bounding_box(page_width, page_height, region_node)
      cells_node = table_structure_node.find('region')
      cells = [self._get_cell(page_width, page_height, node) for node in cells_node]

      yield page_number, Table(table_id, table_rect, cells)

  def _get_bounding_box(self, page_width, page_height, xml_node):
    bounding_box_node = xml_node.find('bounding-box')
    left = self._to_int(bounding_box_node.get('x1'))
    top = page_height - self._to_int(bounding_box_node.get('y2'))
    right = self._to_int(bounding_box_node.get('x2'))
    bottom = page_height - self._to_int(bounding_box_node.get('y1'))
    assert 0 <= left and left < right and right <= page_width
    assert 0 <= top and top < bottom and bottom <= page_height
    return Rect(left, top, right, bottom)

  def _to_int(self, str):
    result = str.replace('ß', '6')
    return int(result)

  def _get_cell(self, page_width, page_height, xml_node):
    text_rect = self._get_bounding_box(page_width, page_height, xml_node)
    col_start = int(xml_node.get('start-col'))
    col_end = int(xml_node.get('end-col', col_start))
    row_start = int(xml_node.get('start-row'))
    row_end = int(xml_node.get('end-row', row_start))
    assert col_start <= col_end and row_start <= row_end
    grid_rect = Rect(col_start, row_start, col_end + 1, row_end + 1)
    return Cell(text_rect, grid_rect)

  def _image_to_byte_array(self, image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


class IcdarSplit(IcdarBase):
  """DatasetBuilder for training SPLIT model."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Generate markup table.'
  }

  def _get_features_dict(self):
    return tfds.features.FeaturesDict({
      'image': tfds.features.Image(shape=(None, None, 3)),
      'horz_split_points_mask': tfds.features.Tensor(shape=(None,), dtype=tf.bool),
      'vert_split_points_mask': tfds.features.Tensor(shape=(None,), dtype=tf.bool),
      # Ground truth table
      'markup_table': tfds.features.Tensor(shape=(), dtype=tf.string)
    })

  def _get_single_example_dict(self, table_image, markup_table):
    """Returns dict with nessary inputs for the model."""

    horz_split_points_mask = markup_table.create_horz_split_points_mask()
    vert_split_points_mask = markup_table.create_vert_split_points_mask()
    return {
      'image': self._image_to_byte_array(table_image),
      'horz_split_points_mask': horz_split_points_mask,
      'vert_split_points_mask': vert_split_points_mask,
      'markup_table': markup_table.to_tensor().numpy()
    }


class IcdarMerge(IcdarBase):
  """DatasetBuilder for training MERGE model."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Generate markup table.'
  }

  def __init__(self, split_checkpoint_path='checkpoints/split_icdar.ckpt', **kwargs):
    super().__init__(**kwargs)
    self._split_checkpoint_path = split_checkpoint_path
    # Lazy initialization
    self._split_model = None

  def _get_features_dict(self):
    return tfds.features.FeaturesDict({
      'image': tfds.features.Image(shape=(None, None, 3)),
      # SPLIT model outputs
      'horz_split_points_probs': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
      'vert_split_points_probs': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
      'horz_split_points_binary': tfds.features.Tensor(shape=(None,), dtype=tf.int32),
      'vert_split_points_binary': tfds.features.Tensor(shape=(None,), dtype=tf.int32),  
      # Ground truth masks    
      'merge_right_mask': tfds.features.Tensor(shape=(None, None), dtype=tf.bool, encoding='zlib'),
      'merge_down_mask': tfds.features.Tensor(shape=(None, None), dtype=tf.bool, encoding='zlib'),
      # Ground truth table
      'markup_table': tfds.features.Tensor(shape=(), dtype=tf.string)
    })

  def _get_single_example_dict(self, table_image, markup_table):
    """Returns dict with nessary inputs for the model."""

    h_probs, v_probs, h_binary, v_binary = self._get_split_model_outputs(table_image)
    grid = GridStructureBuilder(markup_table.rect, h_binary, v_binary).build()
    merge_right_mask, merge_down_mask = markup_table.create_merge_masks(grid)
    return {
      'image': self._image_to_byte_array(table_image),
      'horz_split_points_probs': h_probs,
      'vert_split_points_probs': v_probs,
      'horz_split_points_binary': h_binary,
      'vert_split_points_binary': v_binary,
      'merge_right_mask': merge_right_mask,
      'merge_down_mask': merge_down_mask,
      'markup_table': markup_table.to_tensor().numpy()
    }

  def _get_split_model_outputs(self, table_image):
    table_image_array = tf.keras.utils.img_to_array(
      table_image, data_format='channels_last', dtype='uint8')
    table_image_tensor = tf.convert_to_tensor(table_image_array, dtype='uint8')
    table_image_tensor = tf.expand_dims(table_image_tensor, axis=0)
    outputs_dict = self._get_split_model()(table_image_tensor)
    keys_of_interest = [
      'horz_split_points_probs3', 
      'vert_split_points_probs3',
      'horz_split_points_binary',
      'vert_split_points_binary'
    ]
    return tuple(
      tf.squeeze(outputs_dict[key], axis=0).numpy() for key in keys_of_interest
    )

  def _get_split_model(self):
    if self._split_model is not None:
      return self._split_model

    assert tf.io.gfile.exists(self._split_checkpoint_path)
    model = split.evaluation.load_model(self._split_checkpoint_path)
    
    self._split_model = model
    return model
    