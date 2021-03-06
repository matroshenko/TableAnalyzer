"""ICDAR dataset."""

import tensorflow_datasets as tfds

import context
from datasets.FinTabNet.FinTabNet import FinTabNetSplit


class FinTabNetSplitTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for FinTabNet dataset."""
  DATASET_CLASS = FinTabNetSplit
  SPLITS = {
      'val': 17  # Number of fake train example
  }
  SKIP_TF1_GRAPH_MODE = True
  SKIP_CHECKSUMS = True


if __name__ == '__main__':
  tfds.testing.test_main()
