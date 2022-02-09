# SPLERGE_via_TF
Implementation of SPLERGE model for table structure recognition via TensorFlow. For reference please check original paper: 

C. Tensmeyer, V. I. Morariu, B. Price, S. Cohen and T. Martinez, "Deep Splitting and Merging for Table Structure Decomposition," 2019 International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 114-121, doi: 10.1109/ICDAR.2019.00027.

# Usage
To train SPLIT model use script `train_split_model.py`.

To train MERGE model use script `train_merge_model.py`.

Run with `--help` argument to view usage info.

# Results

As the main metric we used ajacency F-score (see [article](https://www.researchgate.net/publication/233954637_A_Methodology_for_Evaluating_Algorithms_for_Table_Understanding_in_PDF_Documents) for more details).
We have trained our models on GPU (NVIDIA GeForce GTX 1080 Ti) for 100 epochs and obtained following results:

|                   | Train          | Test   |
| ----------------- | ---------------|--------|
| SPLIT             | 0.8863         | 0.5267 |
| SPLIT + MERGE     | 0.8982         | 0.5295 |
| Num of tables     | 81             | 148    |

# Images

![](images/merge_model_predictions.png)
Fig 1. SPLIT + MERGE model predictions on training set.

# Conclusion

More training data will be helpful to decrease the gap between train and test error.