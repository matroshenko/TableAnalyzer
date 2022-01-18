"""ICDAR dataset."""

from collections import namedtuple
import tensorflow_datasets as tfds
from itertools import chain
import os

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
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(
      'https://www.tamirhassan.com/html/files/icdar2013-competition-dataset-with-gt.zip')

    return {
        'train': chain(
          self._generate_examples(path / 'competition-dataset-eu'),
          self._generate_examples(path / 'competition-dataset-us'))
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(ICDAR): Yields (key, example) tuples from the dataset
    for pdf_file_name in path.glob('*.pdf'):
      file_name = os.path.splitext(pdf_file_name)[0]
      print( file_name )
      yield file_name, {
        'image': None,
        'horz_split_points': None,
        'vert_split_points': None
      }


Rect = namedtuple('Rect', ['left', 'top', 'right', 'bottom'])

class Cell(Rect):
  def __init__(self, left, top, right, bottom, col_start, col_end, row_start, row_end):
    super().__init__(left, top, right, bottom)
    self.col_start = col_start
    self.col_end = col_end
    self.row_start = row_start
    self.row_end = row_end

class Table(Rect):
  def __init__(self, left, top, right, bottom, cells):
    super().__init__(left, top, right, bottom)
    self.cells = cells

