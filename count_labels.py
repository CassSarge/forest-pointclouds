import tensorflow as tf
import numpy as np
from test_models import prepare_data
import pickle
import matplotlib.pyplot as plt


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

    # Save history as a list of lists
    occurance_list = [[d.get(i, 0) for i in range(4)] for d in occurance_list]
    with open('./logs/occurance_lists/occurance_list_test_{}'.format(window_width), 'wb') as f:
       pickle.dump(occurance_list, f)

def plot_label_occurances(window_width):
    with open('./logs/occurance_lists/occurance_list_test_{}'.format(window_width), 'rb') as f:
        occurance_list = pickle.load(f)

    occurance_list = np.array(occurance_list)
    occurance_list = np.mean(occurance_list, axis=0)

    # Plot the results
    fig, ax = plt.subplots()
    ax.bar(np.arange(4), occurance_list)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Foliage', 'Stem', 'Ground', 'Undergrowth'])
    ax.set_ylabel("Percent of points")
    ax.set_xlabel("Label")
    ax.set_title("Occurance of Labels in Test Data with Window Width {}m".format(window_width.replace('_', '.')))
    plt.show()

def plot_all_label_means(window_widths):
    mean_list = []
    window_nums= [float(num.replace('_', '.')) for num in window_widths]


    for window_width in window_widths:
        with open('./logs/occurance_lists/occurance_list_test_{}'.format(window_width), 'rb') as f:
            occurance_list = pickle.load(f)
            # Get the mean of each label
            occurance_list = np.array(occurance_list)
            occurance_list = np.mean(occurance_list, axis=0)
            mean_list.append(occurance_list)

    mean_list = np.array(mean_list).T.tolist()

    # Plot the results of all the means as 4 lines on the same plot with different colours
    fig, ax = plt.subplots()
    # Plot the first line
    ax.plot(window_nums, mean_list[0], color='lime')
    # Plot the second line
    ax.plot(window_nums, mean_list[1], color='red')
    # Plot the third line
    ax.plot(window_nums, mean_list[2], color='blue')
    # Plot the fourth line
    ax.plot(window_nums, mean_list[3], color='cyan')
    ax.set_xticks(window_nums)
    ax.set_xticklabels(window_nums)
    ax.set_ylabel("Percent of points")
    ax.set_xlabel("Window width (m)")
    ax.set_title("Occurance of Labels in Test Data")
    ax.legend(['Foliage', 'Stem', 'Ground', 'Undergrowth'], loc='upper left', bbox_to_anchor=(0.65, 0.9))
    plt.show()

    


if __name__ == "__main__":
    batch_size = 16

    window_widths = ["0_5", "1_0", "1_5", "2_0", "2_5", "3_0", "3_5", "4_0", "4_5"]

    for window_width in window_widths:
    #     print("Window width: {}".format(window_width))
    #     filename = 'data/test_data/testing_data_{}.tfrecord'.format(window_width)
    #     count_labels(filename, batch_size, window_width)
        plot_label_occurances(window_width)

    # plot_all_label_means(window_widths)