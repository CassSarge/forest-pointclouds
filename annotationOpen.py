import os
import pickle
import numpy as np

import tensorflow as tf

with open("data/plot_annotations.p", "rb") as f:
    annotations = pickle.load(f)
data = np.asarray(annotations)
print(data.shape) # (10829404, 4)
print(type(data))
features = data[:, 0:3] # Features between 
labels = data[:, 3] # Label between 0 and 3 (4 classes)
print(features.shape)
print(labels.shape)
print(features[1])
print(labels[1:50])

print(type(features))
print(type(features[1]))
print(type(features[1][1]))

print(type(labels))
print(type(labels[1]))


# print("Creating TFRecord file...")
# with tf.io.TFRecordWriter("plot_annotations.tfrecord") as writer:
#     # Make TFRecord file from numpy array
#     for i in range(len(features)):

#         # Create a feature
#         feature = {
#             'points': tf.train.Feature(float_list=tf.train.FloatList(value=features[i])),
#             'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
#         }
#         # Create an example protocol buffer
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
#         # Serialize to string and write on the file
#         writer.write(example.SerializeToString())