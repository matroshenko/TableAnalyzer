# SPLERGE_via_TF
Implementation of SPLERGE model for table structure recognition via TensorFlow. For reference please check original paper: 
C. Tensmeyer, V. I. Morariu, B. Price, S. Cohen and T. Martinez, "Deep Splitting and Merging for Table Structure Decomposition," 2019 International Conference on Document Analysis and Recognition (ICDAR), 2019, pp. 114-121, doi: 10.1109/ICDAR.2019.00027.

# Usage

train.py \<ModelType> \<Dataset> \<PathToResultModelFile>

* \<ModelType> -- type of model, should be SPLIT or MERGE.
* \<Dataset> -- name of the training dataset. Currently only ICDAR dataset is supported.
* \<PathToResultModelFile> -- location of the file, where result model should be serialized.

predict.py \<PathToSplitModelFile> \<PathToMergeModelFile> \<PathToTestImage> \<PathToResultImage>

* \<PathToSplitModelFile> -- path to a previously trained SPLIT model file.
* \<PathToMergeModelFile> -- path to a previously trained MERGE model file.
* \<PathToTestImage> -- path to the file, containing image of a test table.
* \<PathToResultImage> -- path to the file, where resulting image will be saved.
