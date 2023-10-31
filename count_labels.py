import tensorflow as tf
import numpy as np
from test_models import prepare_data
import pickle


def count_labels(filename, batch_size, window_width):
    # Load the dataset
    dataset_train= tf.data.TFRecordDataset(filename)

    # Prepare dataset
    dataset_train = prepare_data(dataset_train, batch_size)

    # Number of points in each window
    num_points = 8192

    # class_labels = {
    #     0: 'Foliage',
    #     1: 'Stem',
    #     2: 'Ground',
    #     3: 'Undergrowth'
    # }

    train_length = dataset_train.reduce(0, lambda x, _: x + 1)
    print("Train length: {}".format(train_length))

    occurance_list = []

    for points, labels in dataset_train:
        # Count how many times each label occurs in the batch
        for label_list in labels: # Loop through the 16 windows in the batch
            label_list = label_list.numpy().astype(int).squeeze()
            # print("Label list: {}, length: {}".format(label_list, len(label_list)))
            unique, counts = np.unique(label_list, return_counts=True)
            percent = np.round(counts / num_points * 100, 2)

            # print("Occurances %: {}".format(dict(zip(unique, percent))))
            occurance_list.append(dict(zip(unique, percent)))

    # print("Occurances %: {}".format(occurance_list))
    # Save history
    with open('./logs/occurance_lists/occurance_list_test_{}'.format(window_width), 'wb') as f:
       pickle.dump(occurance_list, f)




if __name__ == "__main__":
    batch_size = 16

    window_widths = ["0_5", "1_0", "1_5", "2_0", "2_5", "3_0", "3_5", "4_0", "4_5"]

    for window_width in window_widths:
        print("Window width: {}".format(window_width))
        filename = 'data/test_data/testing_data_{}.tfrecord'.format(window_width)
        count_labels(filename, batch_size, window_width)