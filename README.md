# DIST (Denoise and Impute Spatial Transcriptomics)


## Introduction
Spatially resolved transcriptomics technologies enable comprehensive measurement of gene expression patterns in the context of intact tissues. However, existing technologies suffer from either low resolution or shallow sequencing depth. So, we present **DIST**, a deep learning-based method that enhance spatial transcriptomics data by self-supervised learning or transfer learning. Through **self-supervised learning**, DIST can impute the gene expression levels at unmeasured area accurately and improve the data quality in terms of total counts and percentage of dropout. Moreover, **transfer learning** enables DIST improve the imputed gene expressions by borrowing information from other high-quality data. 

With DIST, you can: 

- Impute gene expression profiles on unmeasured spots to improve resolution.
- Improve imputed gene expression by transfer learning.
- Denoise imputed gene expression, as some technical factors lead to substantial noise in sequencing.

DIST mainly focuses on array-based spatial transcriptomic techniques such as matrix-arrayed ST and honeycomb-arrayed Visium.
DIST mainly focuses on array-based spatial transcriptomic techniques. Measured spots of these techniques arrange in certain patterns, generally comprising two classes: matrix arrangement such as ST and honeycomb arrangement such as Visium. 

## Install
To install `DIST` package you must make sure that your `tensorflow` version `1.x`. You decide to use CPU or GPU to run tensorflow according your devices. GPU could accelerate tensorflow by installing `tensorflow-gpu`. In addtation, please make sure your `python` version is compatible with `tensorflow 1.x`. In our paper, we used `python 3.6.x` , `tensorflow-gpu 1.12.x` and `numpy 1.16.x`.

To download and install, open the terminal and use pip:
```
wget https://github.com/zhaoyp1997/DIST/raw/main/DIST-1.0-py3-none-any.whl
pip install DIST-1.0-py3-none-any.whl
```
Or change to a directory where you want DIST to be downloaded to and do:
```
git clone https://github.com/zhaoyp1997/DIST.git
cd DIST
python setup.py install
```
Depending on your user privileges, you may have to add `--user` as an argument to `setup.py`. 


## Usage
DIST algorithm is based on neural network framework, so it contains training and testing processes. We need to create training set and test set from spatial transcriptomic data. Here, self-supervised learning means training and test set come from the same spatial gene expression, while transfer learning means two different sources and training source is more informative.

Data we need contain spatial **gene expression matrix** and matched **spot coordinate matrix** for both training set and test set. For gene expression matrix, each row corresponds to a spot with a barcode, and each column corresponds to a gene with a gene id. For spot coordinate matrix, each row corresponds to a spot with a barcode, the first column corresponds to row coordinates of the spots and the second column corresponds to column coordinates of the spots. These two matrices have the same number of rows, their values with the same row index describe the same spot.

Next, we display several key functions to simply describe the usage of DIST. For further details, please see below jupyter notebook examples.

---
Firstly, import DIST package.
```python
from DIST import *
```

Secondly, create training and test dataset from expression and cooridate matrix; get information about imputed coordinates for transforming outputs of DIST network into expression and coordinates matrices.

```python
# For matrix-arrayed spatial transcriptomic data represented by ST.
train_set = getSTtrainset(train_counts, train_coords)
test_set = getSTtestset(test_counts, test_coords)
position_info = get_ST_position_info(integral_coords)
```
or
```python
# For honeycomb-arrayed spatial transcriptomic data represented by Visium.
train_set = get10Xtrainset(train_counts, train_coords)
test_set = get10Xtestset(test_counts, test_coords)
position_info = get_10X_position_info(integral_coords)
```
where `train_counts` is a 2-D array or dataframe of gene expression from train data. Each row corresponds to a spot with a barcode, and each column corresponds to a gene with a gene id. `train_coords` is a 2-D ndarray or dataframe of spots coordinates from train data. Each row corresponds to a spot with a barcode, the first column corresponds to row coordinates of the spots and the second column corresponds to column coordinates of the spots. 

`train_counts` and `train_coords` have the same number of rows, their values with the same row index describe the same spot. `test_counts` and `test_coords` have similar explanation but from test data.

`intergral_coords` describes row and column coordinates of some original spots that are expected to preserve. Usually, it contains the initial spots of test data without quality control for tissue structural integrity.

Thirdly, run DIST; transform the outputs of network into imputed expression matrix and its spot coordinate matrix.
```python
imputed_img = DIST(train_set, test_set, epoch=200, batch_size=512, learning_rate=0.001, gpu='0')
imputed_counts, imputed_coords = img2expr(imputed_img, gene_ids, intergal_coords, position_info)
```
where `epoch`, `batch_size`, `learning_rate` are parameters of network, `gpu` is the GPU or GPUs you want to use.

---

We also provide two examples for reproducing the results of ST melanoma and 10X Visium human invasive ductal carcinoma (IDC) in our paper. These two examples use respectively self-supervised learning and transfer learning.

- For ST melanoma and self-supervised learning, please refer to [a self-supervised learning example for ST human melanoma (mel1_rep1) data.ipynb](./a_self-supervised_learning_example_for_ST_human_melanoma_(mel1_rep1)_data.ipynb).
- For Visium IDC data and transfer learning, please refer to [a transfer-learning example for Visium human invasive ductal carcinoma (IDC) data.ipynb](./a_transfer_learning_example_for_Visium_human_invasive_ductal_carcinoma_(IDC)_data.ipynb).

## License
DIST is licensed under the Apache License 2.0.

## Citation
