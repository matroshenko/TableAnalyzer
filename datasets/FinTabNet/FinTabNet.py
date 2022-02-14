"""FinTabNet table recognition dataset."""

from abc import abstractmethod
import xml.etree.ElementTree as ET
import io
import os
import glob
import pathlib
import json

import tensorflow_datasets as tfds
import tensorflow as tf
import pdf2image
from PyPDF2 import PdfFileReader

from datasets.ICDAR.markup_table import Cell, Table
from utils.rect import Rect
from table.grid_structure import GridStructureBuilder
from split.model import Model


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


class FinTabNetBase(tfds.core.GeneratorBasedBuilder):
  """Base DatasetBuilder for FinTabNet datasets."""

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=self._get_features_dict(),
        homepage='https://developer.ibm.com/exchanges/data/all/fintabnet/',
        citation=_CITATION,
        disable_shuffling=True
    )

  @abstractmethod
  def _get_features_dict(self) -> tfds.features.FeaturesDict:
    """Returns features, describing dataset element."""
    pass

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    path = dl_manager.download_and_extract(
      'https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/fintabnet.tar.gz')

    return {
        'train': self._generate_examples(path, 'train'),
        'val': self._generate_examples(path, 'val'),
        'test': self._generate_examples(path, 'test')
    }

  def _generate_examples(self, path, split):
    """Yields examples for specified split."""

    jsonl_file_name = os.path.join(path, 'FinTabNet_1.0.0_table_example.jsonl')
    with open(jsonl_file_name, 'r') as f:
      for line in f:
        sample = json.loads(line)
        if sample['split'] != split:
          continue

        pdf_file_name = os.path.join(path, 'pdf', sample['filename'])
        pdf_height, pdf_width = self._get_pdf_file_shape(pdf_file_name)

        table_rect = self._bbox_to_rect(pdf_height, sample['bbox'])
        table_image = self._get_table_image(pdf_file_name, table_rect)

        cells = self._get_markup_cells(pdf_height, sample['html']['structure']['tokens'], sample['cells'])
        table_id = sample['table_id']
        table = Table(table_id, table_rect, cells)
        yield table_id, self._get_single_example_dict(table_image, table)

  @abstractmethod
  def _get_single_example_dict(self, table_image, markup_table):
    """Returns dict with nessary inputs for the model."""
    pass

  def _get_pdf_file_shape(self, pdf_file_name):
    with open(pdf_file_name, 'rb') as pdf_file:
      pdf_page = PdfFileReader(pdf_file).getPage(0)
      pdf_shape = pdf_page.mediaBox
      pdf_height = pdf_shape[3]-pdf_shape[1]
      pdf_width = pdf_shape[2]-pdf_shape[0]
    return pdf_height, pdf_width

  def _bbox_to_rect(self, page_height, bbox):
    return Rect(
      int(bbox[0]),
      int(page_height - bbox[3]),
      int(bbox[2]),
      int(page_height - bbox[1])
    )

  def _get_table_image(self, pdf_file_name, table_rect):
    pdf_height, pdf_width = self._get_pdf_file_shape(pdf_file_name)
    page = pdf2image.convert_from_path(pdf_file_name, size=(pdf_width, pdf_height))[0]
    return page.crop(table_rect.as_tuple())

  def _get_markup_cells(self, page_height, html_tokens, cells):
    # TODO
    pass

  def _image_to_byte_array(self, image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


class FinTabNetSplit(FinTabNetBase):
  """DatasetBuilder for training SPLIT model."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.'
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


class FinTabNetMerge(FinTabNetBase):
  """DatasetBuilder for training MERGE model."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.'
  }

  def __init__(self, split_checkpoint_path='checkpoints/split.ckpt', **kwargs):
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
    # Split model can't run in graph mode.
    assert tf.executing_eagerly()
    model = Model()
    random_image = tf.random.uniform(shape=(1, 100, 200, 3), minval=0, maxval=255, dtype='int32')
    model(random_image)
    model.load_weights(self._split_checkpoint_path)
    
    self._split_model = model
    return model
    