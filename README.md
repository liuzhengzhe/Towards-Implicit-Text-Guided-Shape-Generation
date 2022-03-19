# Towards Implicit Text-Guided 3D Shape Generation
## Towards Implicit Text-Guided 3D Shape Generation (CVPR2022)

Code for the paper [Towards Implicit Text-Guided 3D Shape Generation], CVPR 2022.

This code is based on IM-Net https://github.com/czq142857/IM-NET-pytorch

**Authors**: Zhengzhe Liu, Xiaojuan Qi, Yi Wang, Chi-Wing Fu

<img src="image.PNG" width="900"/>


## Installation

### Requirements
* Python 3.8.8
* Pytorch 1.10.0
* CUDA 11.3
* h5py
* scipy
* [mcubes](https://github.com/pmneila/PyMCubes)
* pytorch_lamb


## Data Preparation

* Download our [hdf5_train_new]() and [hdf5_test_new]().

OR

* Download the [dataset](http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_128_solid.zip).

* unzip it to "generator" folder.

```
python 2_gather_256vox_16_32_64.py.py 
```


## Pretrained Model
We provide pretrained models for each training step. Download it [here](). "Put them to generation/checkpoint"



## Inference

(1) Text-Guided Shape Generation

```
python main.py --res64 --sample_dir samples/im_ae_out --start 0 --end 7454 --high_resolution
```

You can generate coarse shapes fast by removing "--high_resolution"

(2) Diversified Generation

```
python main.py --div --sample_dir samples/im_ae_out --start 0 --end 7454 --high_resolution
```

Others:

(1) Auto-Encoder

```
python main.py --ae --sample_dir samples/im_ae_out --start 0 --end 7454
```

## Training Generation Model

```
sh train.sh
```

## Manipulation

Coming soon. 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (liuzhengzhelzz@gmail.com).
