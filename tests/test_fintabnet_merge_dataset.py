"""ICDAR dataset."""

from functools import partial
import os

import tensorflow_datasets as tfds

import context
from datasets.FinTabNet.FinTabNet import FinTabNetMerge


class FinTabNetSplitTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for FinTabNet dataset."""
  DATASET_CLASS = FinTabNetMerge
  SPLITS = {
      'train': 17  # Number of fake train example
  }
  # Split model can't run in graph mode.
  SKIP_TF1_GRAPH_MODE = True
  SKIP_CHECKSUMS = True

  def _make_builder(self, config=None):
    return self.DATASET_CLASS(  # pylint: disable=not-callable
        split_checkpoint_path=os.path.join(self.dummy_data, 'split.ckpt'),
        data_dir=self.tmp_dir,
        config=config,
        version=self.VERSION)


if __name__ == '__main__':
  tfds.testing.test_main()
