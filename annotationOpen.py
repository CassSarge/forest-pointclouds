import os
import pickle
import numpy as np

import tensorflow as tf

# Attempting to create a tfrecord file from the plot annotations
# This might be the wrong way to go about it! because all of the points are grouped together rather than being in mini plots that can be split up idk
def create_example(features, labels):

    

	feature = {
        'scan/points' : tf.train.Feature(float_list=tf.train.FloatList(value=features[:, :3].reshape(-1, 1))),
        'label/labels' : tf.train.Feature(float_list=tf.train.FloatList(value=[labels]))
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))

if __name__ == '__main__':

    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    print(data.shape) # (10829404, 4) (x, y, z, label)
    print(type(data)) # numpy.ndarray
    features = data[:, 0:3] # Features between 
    labels = data[:, 3] 
    print(features.shape) # (10829404, 3)
    print(labels.shape) # (10829404,)
    print(features[1, :3]) #
    print(labels[1:50])

    print(type(features))
    print(type(features[1]))
    print(type(features[1][1]))

    print(type(labels))
    print(type(labels[1]))

    # create_example(features, labels)

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