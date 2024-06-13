# forest-pointclouds

Uses Pointnet++ as implemented by [dgriffiths3](https://github.com/dgriffiths3/pointnet2-tensorflow2) in order to analyse how changing the size of the xy-window used in semantic segmentation of a large forest point cloud impacts overall results. My contributions were made in service of my thesis ["Forest for the Trees: Semantic Segmentation of LiDAR Point Clouds in Complex Forest Environments"](https://drive.google.com/file/d/18cdgXkR1VnELQJAbTLLXP87Gec_HpgJy/view?usp=sharing), for my honours undergraduate in Mechatronics Engineering, Majoring in Space Engineering. This thesis was produced with guidance from Dr Mitch Bryson and would not have been possible without his work.

## Requirements and Setup

To use this repo you must first select versions of CUDA, cuDNN, and Tensorflow that are compatible with your graphics card, operating system and this repository. This can be extremely challenging to find. I used a Nvidia GeForce RTX 2070, CUDA 11.2, cuDNN 8.1, Tensorflow 2.11, on Ubuntu 18.04.6.

Some of the code in `tf_ops` will also have to be changed to reflect your version, such as tf_ops/compile_ops.sh line 5 and all Status() calls in tf_ops/3d_interpolation/tf_interpolate.cpp. Changes between this repo and dgriffiths3 original implementation should reflect which lines were changed moving from CUDA 10.1 to 11.2. My data was stored in pickle files (.p) in row entries of x,y,z,label where a label could be 0-3.

As with dgriffiths3 implementation,
> To run the ModelNet or ScanNet example first download the tfrecords containing the training data from [here](https://drive.google.com/drive/folders/1v5B68RHgDI95KM4EhDrRJxLacJAHcoxz) and place in a folder called data. To compile the tensorflow Ops first ensure the CUDA_ROOT path in tf_ops/compile_ops.sh points correctly to you cuda folder then compile the ops with:
```
chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
```
## Usage

For testing the setup is correct and everything is installed:
```
conda activate tf
python train_scannet.py
```
For real usage
```
conda activate tf
python gen_and_run.py
```
and respond to input prompt with a window size (e.g '4')

## Logs
All logs are saved to the log directory corresponding to the window size used, e.g. 'logs/trees_4' for a window size of 4m

Information can be visualised in tensorboard with `tensorboard --logdir=logs/trees_N` , which is then viewed in the web browser at *http://localhost:6006*
