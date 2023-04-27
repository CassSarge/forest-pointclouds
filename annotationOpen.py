import os
import pickle
import numpy as np
import torch
import torch.utils as utils

with open("plot_annotations.p", "rb") as f:
    annotations = pickle.load(f)
data = np.asarray(annotations)
print(data.shape) # (10829404, 4)
print(type(data))
features = data[:, 0:3] # Features between 
print(features.shape)
labels = data[:, 3] # Label between 0 and 3 (4 classes)
print(labels.shape)
# print(labels[0:100])
print(features[1])

train_x = torch.Tensor(features)
train_y = torch.Tensor(labels)

dataset = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
