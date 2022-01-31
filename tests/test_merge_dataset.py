"""ICDAR dataset."""

from functools import partial
import os

import tensorflow_datasets as tfds

import context
from datasets.ICDAR.ICDAR import IcdarMerge


class IcdarSplitTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for ICDAR dataset."""
  DATASET_CLASS = IcdarMerge
  SPLITS = {
      'train': 15  # Number of fake train example
  }
  # Split model can't run in graph mode.
  SKIP_TF1_GRAPH_MODE = True

  def _make_builder(self, config=None):
    return self.DATASET_CLASS(  # pylint: disable=not-callable
        split_checkpoint_path=os.path.join(self.dummy_data, 'split.ckpt'),
        data_dir=self.tmp_dir,
        config=config,
        version=self.VERSION)


if __name__ == '__main__':
  tfds.testing.test_main()
