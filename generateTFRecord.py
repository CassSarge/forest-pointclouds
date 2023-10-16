import tensorflow as tf
import numpy as np
import pickle

def load_dataset(split: bool = True):
    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    features = data[:, 0:3] # x, y, z
    labels = data[:, 3] # label

    if split:
        return features, labels
    else:
        return data

def load_test_dataset(split: bool = True):
    with open("data/plot_annotations_HQPLR118V1.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    features = data[:, 0:3] # x, y, z
    labels = data[:, 3] # label

    if split:
        return features, labels
    else:
        return data

def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def example_from_data(features, labels, point_start: int, point_end: int):
     
    feature = {
		'points' : float_list_feature(np.squeeze(features[point_start:point_end, :3].reshape(-1, 1))),
		'labels' : float_list_feature(np.squeeze(labels[point_start:point_end].reshape(-1, 1))),
	}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

def basic_example_from_data(features, labels):
     
    feature = {
		'points' : float_list_feature(np.squeeze(features[:, :3].reshape(-1, 1))),
		'labels' : float_list_feature(np.squeeze(labels[:].reshape(-1, 1))),
	}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

def generate_training_set(features, labels, example_size: int, training_end: int):
    features = features[:training_end,:]
    labels = labels[:training_end]
    
    print("Creating training TFRecord file...")
    
    with tf.io.TFRecordWriter("data/plot_annotations_training.tfrecord") as writer:
        for i in range(int(training_end // example_size)):
            print("Writing training example {} of {}".format(i, int(training_end // example_size)))
            start_pt = i * example_size
            end_pt = start_pt + example_size
            
            example = example_from_data(features, labels, start_pt, end_pt)
            writer.write(example.SerializeToString())


    print("Done!")

def generate_validation_set(features, labels, example_size: int, training_end: int, validation_end: int):
    features = features[training_end:validation_end,:]
    labels = labels[training_end:validation_end]
    data_length = (validation_end - training_end )

    print("Creating validation TFRecord file...")
    
    with tf.io.TFRecordWriter("data/plot_annotations_validation.tfrecord") as writer:
        for i in range(int(data_length// example_size)):
            print("Writing validation example {} of {}".format(i, int(data_length // example_size)))
            start_pt = i * example_size
            end_pt = start_pt + example_size
            
            example = example_from_data(features, labels, start_pt, end_pt)
            writer.write(example.SerializeToString())

    print("Done!")


if __name__ == '__main__':

    features, labels = load_dataset()

    example_size = 100000
    training_end = 8000000
    validation_end = 10000000

    generate_training_set(features, labels, example_size, training_end)

    generate_validation_set(features, labels, example_size, training_end, validation_end)
