import tensorflow as tf
import os
from visualisation import create_model
import tensorflow as tf
# from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
import pickle

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

    # Create the model
    model = create_model(config)

    # Define the directory containing the test data
    test_data_dir = '/home/cass/thesis/forest-pointclouds/'

    # Define log folders where models are stored
    checkpoint_names = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    checkpoint_dirs = ["./logs/trees_{}/model/weights'".format(name) for name in checkpoint_names]

    # Define the list of test data files
    test_data_nums = ['0_5', '1_0', '1_5', '2_0', '2_5', '3_0', '3_5', '4_0', '4_5']
    test_data_dirs = ["./data/test_data/testing_data_{}.tfrecord".format(num) for num in test_data_nums]

    # Loop through each model and test data file, and evaluate the model
    for i in range(checkpoint_dirs):
        print("Testing model {}".format(checkpoint_names[i]))
        
        # Load the model
        model.load_weights(checkpoint_dirs[i]).expect_partial()

        # Load the test data
        test_data = tf.data.TFRecordDataset(test_data_dirs[i])

        # Evaluate the model on the test data
        history = model.evaluate(test_data)
        
        # Save history
        with open('./logs/trees_{}/testHistoryDict'.format(checkpoint_names[i]), 'wb') as f:
            pickle.dump(history.history, f)

        # Print the accuracy and meanIoU
        print("Model {}, accuracy: {:5.2f}%".format(checkpoint_names[i], 100 * history["sparse_cat_acc"]))                        
        print("Model {}, meanIoU: {:5.2f}".format(checkpoint_names[i], history["meanIoU"]))

