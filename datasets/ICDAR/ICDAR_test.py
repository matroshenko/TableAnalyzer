"""ICDAR dataset."""

import tensorflow_datasets as tfds
import ICDAR


class IcdarTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for ICDAR dataset."""
  # TODO(ICDAR):
  DATASET_CLASS = ICDAR.Icdar
  SPLITS = {
      'train': 7  # Number of fake train example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
