import tensorflow as tf
from visualisation import create_model
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def plot_confusion_matrix(confmat, title='Confusion matrix'):

    con_mat_norm = np.around(confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis], decimals=2)
    
    con_mat_df = pd.DataFrame(con_mat_norm)

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111)
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    names = ['Foliage', 'Stem', 'Ground', 'Undergrowth']
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    # Set title
    plt.title(title)
    plt.show()



def prepare_data(dataset, batch_size):

    n_points = 8192
    
    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'labels': tf.io.FixedLenFeature([n_points], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        labels = sample['labels']

        points = tf.reshape(points, shape=(n_points, 3))
        labels = tf.reshape(labels, shape=(n_points, 1))

        return points, labels

    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def plot_test_results(test_data_nums):
    # Define the order the results appear in
    metric_list = ['loss', 'sparse_cat_acc', 'meanIoU', 'FoliageIoU', 'StemIoU', 'GroundIoU', 'UndergrowthIoU']

    # Define the directory containing the testHistoryDict files
    test_dirs = ['./logs/test_history/testHistory_{}'.format(num) for num in test_data_nums]

    # Create a list of window_widths from test_data_nums
    window_widths = [float(num.replace('_', '.')) for num in test_data_nums]

    # Create empty lists
    accuracies = []
    meanIoUs = []
    losses = []
    FoliageIous = []
    StemIoUs = []
    GroundIoUs = []
    UndergrowthIoUs = []

    # Load the accuracies for each window width from each testHistoryDict file
    for i in range(len(test_dirs)):
        with open(test_dirs[i], 'rb') as f:
            history = pickle.load(f)

            # Append the accuracy and meanIoU to the lists based on their position in metric_list
            accuracies.append(history[metric_list.index('sparse_cat_acc')])
            meanIoUs.append(history[metric_list.index('meanIoU')])
            losses.append(history[metric_list.index('loss')])
            FoliageIous.append(history[metric_list.index('FoliageIoU')])
            StemIoUs.append(history[metric_list.index('StemIoU')])
            GroundIoUs.append(history[metric_list.index('GroundIoU')])
            UndergrowthIoUs.append(history[metric_list.index('UndergrowthIoU')])


    # Plot the accuracy
    plt.plot(window_widths, accuracies, label='Accuracy')
    plt.xlabel('Window Width')
    plt.ylabel('Accuracy')
    plt.title('Accuracy compared for Window Width on Testing Data', fontsize=14)
    plt.grid()
    plt.ylim(0.8, 1)
    plt.show()

    # Plot the IoUs
    plt.plot(window_widths, meanIoUs, label='meanIoU', color='b')
    plt.plot(window_widths, FoliageIous, label='FoliageIoU', color='g')
    plt.plot(window_widths, StemIoUs, label='StemIoU', color='r')
    plt.plot(window_widths, GroundIoUs, label='GroundIoU', color='c')
    plt.plot(window_widths, UndergrowthIoUs, label='UndergrowthIoU', color='m')
    plt.xlabel('Window Width')
    plt.ylabel('IoU')
    plt.title('IoU compared for Window Width on Testing Data', fontsize=14)
    plt.legend()
    plt.grid()
    plt.ylim(0.0, 1)
    plt.show()

    # Plot the loss
    plt.plot(window_widths, losses, label='Loss')
    plt.xlabel('Window Width')
    plt.ylabel('Loss')
    plt.title('Loss compared for Window Width on Testing Data', fontsize=14)
    plt.grid()
    # plt.ylim(0, 1)
    plt.show()

    # Make an empty dictionary to store the stats with the data nums as the keys
    stats = {}

    # Plot confusion matrices
    for i in range(len(test_data_nums)):
        with open('./logs/test_history/confmat_{}'.format(test_data_nums[i]), 'rb') as f:
            confmat = pickle.load(f)
            plot_confusion_matrix(confmat, title='Normalised Confusion Matrix for {}m Window Width'.format(window_widths[i]))
            # Calculate per class TP, FP, FN, TN
            foliage_values = calculate_stats(confmat, 0)
            stem_values = calculate_stats(confmat, 1)
            ground_values = calculate_stats(confmat, 2)
            undergrowth_values = calculate_stats(confmat, 3)
            # Construct dictionary for these values
            values = {'foliage': foliage_values, 'stem': stem_values, 'ground': ground_values, 'undergrowth': undergrowth_values}
            # Add to stats dictionary
            stats[test_data_nums[i]] = values
            # Print the stats
            print("Stats for {}m Window Width:".format(window_widths[i]))
            print("Foliage: {}".format(foliage_values))
            print("Stem: {}".format(stem_values))
            print("Ground: {}".format(ground_values))
            print("Undergrowth: {}".format(undergrowth_values))

    # Save stats dictionary
    with open('./logs/test_history/stats', 'wb') as f:
        pickle.dump(stats, f)

def calculate_stats(confmat, class_num):
    TP = confmat[class_num, class_num]
    FP = np.sum(confmat[:, class_num]) - TP
    FN = np.sum(confmat[class_num, :]) - TP
    TN = np.sum(confmat) - TP - FP - FN

    # Construct a dictionary
    values = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
    return values

def evaluate_models(test_data_nums, config):
    # Create the model
    model = create_model(config)

    # Define log folders where models are stored
    checkpoint_names = [num.replace('_', '.') for num in test_data_nums]
    print(checkpoint_names)
    checkpoint_dirs = ["./logs/trees_{}/model/weights".format(name) for name in checkpoint_names]

    # Define the list of test data files
    test_data_dirs = ["./data/test_data/testing_data_{}.tfrecord".format(num) for num in test_data_nums]

    # Loop through each model and test data file, and evaluate the model
    for i in range(len(checkpoint_dirs)):
        print("Testing model {}".format(checkpoint_names[i]))
        
        # Load the model
        model.load_weights(checkpoint_dirs[i]).expect_partial()

        # Load the test data
        test_data = tf.data.TFRecordDataset(test_data_dirs[i])

        # Prepare dataset
        dataset_test = prepare_data(test_data, config['batch_size'])

        # Pull out the labels
        y_test = []
        y_test = np.concatenate([y for x, y in dataset_test], axis=0)

        print("y shape: {}".format(y_test.shape))

        # Evaluate the model on the test data
        # history = model.evaluate(dataset_test)
        y_pred = model.predict(dataset_test)  # (164, 8192, 4)
        y_pred = np.argmax(y_pred, axis=-1)
        # y_test = np.concatenate(labels, axis=0) # (164, 8192, 1)
        # Check shapes

        # Change y_test from (164, 8192, 1) to (164, 8192)
        y_test = np.squeeze(y_test, axis=-1)

        # Squash down shapes of both to 1D
        y_pred = np.reshape(y_pred, (-1,))
        y_test = np.reshape(y_test, (-1,))

        print("y_pred shape: {}".format(y_pred.shape))
        print("y_test shape: {}".format(y_test.shape))

        # print the first 10 values of each
        print("y_pred: {}".format(y_pred[:10]))
        print("y_test: {}".format(y_test[:10]))

        confmat = tf.math.confusion_matrix(y_test, y_pred, num_classes=4).numpy()
        print(confmat)
        
        # Save history
        with open('./logs/test_history/confmat_{}'.format(test_data_nums[i]), 'wb') as f:
            pickle.dump(confmat, f)

        # # Print the accuracy and meanIoU
        # print("Model {}, accuracy: {:5.2f}%".format(checkpoint_names[i], 100 * history["sparse_cat_acc"]))                        
        # print("Model {}, meanIoU: {:5.2f}".format(checkpoint_names[i], history["meanIoU"]))


    print("Done evaluating models!")


if __name__ == "__main__":

    tf.compat.v1.enable_eager_execution()
    # Set it up such that only 90% of the GPU memory is used, prevents crashing
    tempconfig = tf.compat.v1.ConfigProto()
    tempconfig.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
    tempconfig.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=tempconfig))

    # Parameters for the model and training
    config = {
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 1,
        'num_classes' : 4,
        'lr' : 0.001,
        'bn' : True,
    }


    test_data_nums = ['0_5', '1_0', '1_5', '2_0', '2_5', '3_0', '3_5', '4_0', '4_5']
    # reverse the order
    test_data_nums = test_data_nums[::-1]

    # Evaluate the models
    # evaluate_models(test_data_nums, config)
    plot_test_results(test_data_nums)