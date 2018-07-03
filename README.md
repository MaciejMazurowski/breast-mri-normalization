Pytorch implementation for Image intensity normalization of breast DCE-MRI.


This package includes 3D tissue segmentation for breast DCE-MRI and intensity normalization.
The code was written by Jun Zhang, Department of Radiology at Duke University.

Applications

3D tissue segmentation of breast, dense tissue and heart




Prerequisites

Linux python 2.7
NVIDIA GPU + CUDA CuDNN
Getting Started

Installation

Install pytorch and dependencies from http://pytorch.org/
Install SimpleITK with pip install SimpleITK
Install numpy, scipy, and matplotlib with pip install numpy scipy matplotlib


cd NormCode/Code
Apply our Pre-trained Model with GPU

python main.py  --pre ../TestData/test_pre.nii.gz --post1 ../TestData/test_post1.nii.gz

Apply our Pre-trained Model with CPU
python main.py  --pre ../TestData/test_pre.nii.gz --post1 ../TestData/test_post1.nii.gz --gpu 0


Citation

If you use this code for your research, please cite our paper:


