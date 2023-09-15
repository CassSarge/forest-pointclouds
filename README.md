# forest-pointclouds

Forest 3D Pointcloud Perception and Deep Learning for my undergraduate thesis in Mechatronics Engineering, Majoring in Space Engineering (Honours)

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