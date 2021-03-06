# SPLERGE_via_TF
Implementation of SPLERGE model for table structure recognition via TensorFlow. For reference please check original paper: 

C. Tensmeyer, V. I. Morariu, B. Price, S. Cohen and T. Martinez, "Deep Splitting and Merging for Table Structure Decomposition," 2019 International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 114-121, doi: 10.1109/ICDAR.2019.00027.

# Installation

1. Install poppler for pdf support: `sudo apt install poppler-utils`.
2. Install python packages: `pip install -r requirements.txt`.
3. Build custom ops (see [section](https://www.tensorflow.org/guide/create_op#build_the_op_library) for more info):
```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared ops/*.cpp -o ops/ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

# Usage
To train model use script `train_model.py`.

To evaluate model use script `evaluate_model.py`.

To export SPLERGE model to SavedModel format use script `export_to_saved_model.py`.

Run with `--help` argument to view usage info.

# Datasets

Currently we implemented 2 datasets: ICDAR 2013 and FinTabNet. 

|Dataset|Train size|Test size|
|-|-|-|
|ICDAR|81|148|
|FinTabNet|91220|10629|

# Results

As the main metric we used ajacency F-score (see [article](https://www.researchgate.net/publication/233954637_A_Methodology_for_Evaluating_Algorithms_for_Table_Understanding_in_PDF_Documents) for more details).

We have trained our models on ICDAR dataset for 100 epochs and on FinTabNet dataset for 5 epochs.

|Model|Train dataset|Test dataset|Adj. F-score|
|-|-|-|-|
|SPLIT|ICDAR|ICDAR|0.5267|
|SPLERGE|ICDAR|ICDAR|0.5295|
|SPLIT|FinTabNet|FinTabNet|0.8923|
|SPLERGE|FinTabNet|FinTabNet|0.9345|

# Images

![](images/merge_model_predictions.png)
Fig 1. SPLERGE model predictions on FinTabNet test set.
