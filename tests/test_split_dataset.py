"""ICDAR dataset."""

import tensorflow_datasets as tfds

import context
from datasets.ICDAR.ICDAR import IcdarSplit


class IcdarSplitTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for ICDAR dataset."""
  DATASET_CLASS = IcdarSplit
  SPLITS = {
      'train': 15  # Number of fake train example
  }
  SKIP_CHECKSUMS = True


if __name__ == '__main__':
  tfds.testing.test_main()
