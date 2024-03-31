# forest-pointclouds

Uses Pointnet++ as implemented by [dgriffiths3](https://github.com/dgriffiths3/pointnet2-tensorflow2) in order to analyse how changing the size of the xy-window used in semantic segmentation of a large forest point cloud impacts overall results. My contributions were made in service of my thesis "Forest for the Trees: Semantic Segmentation of LiDAR Point Clouds in Complex Forest Environments", for my honours undergraduate in Mechatronics Engineering, Majoring in Space Engineering. This thesis was produced with guidance from Dr Mitch Bryson and would not have been possible without his work.

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
