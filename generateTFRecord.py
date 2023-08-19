import tensorflow as tf
import numpy as np
import pickle

def load_dataset(data_end):
    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    # Going up to 10 million out of the slightly higher total
    # because it makes it divisible easily
    features = data[:data_end, 0:3] # x, y, z
    labels = data[:data_end, 3] # label

    print(len(features))
    return features, labels

def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def example_from_data(features, labels, point_start: int, point_end: int):
     
    feature = {
		'points' : float_list_feature(np.squeeze(features[point_start:point_end, :3].reshape(-1, 1))),
		'labels' : float_list_feature(np.squeeze(labels[point_start:point_end].reshape(-1, 1))),
	}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

if __name__ == '__main__':
    data_end = 10000000

    features, labels = load_dataset(data_end)

    example_size = 100000
    start_pt = 0
    end_pt = start_pt + example_size
    
    print("Creating TFRecord file...")
    
    with tf.io.TFRecordWriter("data/plot_annotations.tfrecord") as writer:
        for i in range(int(data_end // example_size)):
            print("Writing example {} of {}".format(i, int(data_end // example_size)))
            start_pt = i * example_size
            end_pt = start_pt + example_size
            
            example = example_from_data(features, labels, start_pt, end_pt)
            writer.write(example.SerializeToString())


    print("Done!")