import tensorflow as tf
import numpy as np
import pickle

def load_dataset():
    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    features = data[:, 0:3] # x, y, z
    labels = data[:, 3] # label
    return features, labels

def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == '__main__':
    features, labels = load_dataset()

    print("Generating features...")
    feature = {
		'scan/points' : float_list_feature(np.squeeze(features[:, :3].reshape(-1, 1))),
		'label/points' : float_list_feature(np.squeeze(labels[:].reshape(-1, 1))),
	}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    print("Creating TFRecord file...")
    with tf.io.TFRecordWriter("data/plot_annotations.tfrecord") as writer:
        writer.write(example.SerializeToString())

    print("Done!")