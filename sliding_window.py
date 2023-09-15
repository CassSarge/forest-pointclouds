import numpy as np
import pickle
import tensorflow as tf

from generateTFRecord import load_dataset, basic_example_from_data

def centre_data(data):
    # Centre the data around the origin
    # Find the mean of the x and y coordinates
    mean_x = np.mean(data[:,0])
    mean_y = np.mean(data[:,1])
    # Subtract the mean from the x and y coordinates
    data[:,0] -= mean_x
    data[:,1] -= mean_y

    return data

def normalise_data(data):
    # Normalise the xyz coordinates between -1 and 1
    # Find the max and min of the x,y and z coordinates
    max_x = np.max(data[:,0])
    min_x = np.min(data[:,0])
    max_y = np.max(data[:,1])
    min_y = np.min(data[:,1])
    max_z = np.max(data[:,2])
    min_z = np.min(data[:,2])
    # Normalise the x,y and z coordinates between -1 and 1
    data[:,0] = (((data[:,0] - min_x) / (max_x - min_x)) * 2) - 1
    data[:,1] = (((data[:,1] - min_y) / (max_y - min_y)) * 2) - 1
    data[:,2] = (((data[:,2] - min_z) / (max_z - min_z)) * 2) - 1

    return data

def subsample_to_TFrecord(writer, data, sample_size, n_subsamples, randomizer, replacement = False):

    # Centre the data around the origin
    data = centre_data(data)
    # Normalise the data
    data = normalise_data(data)

    # Repeatedly subsample the data and save it to the TFRecord file
    for i in range(n_subsamples):
        # Subsample the data
        subsampled_data = randomizer.choice(data, sample_size, replace=replacement)
        # Create a TFRecord example from the data
        current_example = basic_example_from_data(subsampled_data[:, 0:3], subsampled_data[:, 3])
        writer.write(current_example.SerializeToString())


    return

def sliding_window(file_location, full_data, window_width: int, sample_num: int, overlap: float= 0.3, sample_size: int = 8192):

    start_x = -20
    start_y = -20

    end_x = 20
    end_y = 20

    current_x = start_x
    current_y = start_y
    
    stride = (1-overlap)*window_width

    i = 0

    rng = np.random.default_rng(12345)
    with tf.io.TFRecordWriter(file_location) as writer:

        while current_y < end_y:

            while current_x < end_x:
                # Get points within window
                current_data = full_data[(full_data[:,0] >= current_x - window_width) & (full_data[:,0] < current_x + window_width) & (full_data[:,1] >= current_y - window_width) & (full_data[:,1] < current_y + window_width)]
                # Get labels within window
                # Count number of points and print
                (num_points, _) = current_data.shape                
                print("i: {}, num_points: {}, Current x: {}, Current y: {}".format(i, num_points, current_x, current_y))
                # Point densities range from about 0 to 40000 within a window right now
                # Choose a random subsample of these
                if num_points > sample_size:
                    subsample_to_TFrecord(writer, current_data, sample_size, sample_num, rng)
                # Else if there are less than 8192 points, but more than 4096, we'll upsample
                elif num_points > sample_size / 2:
                    # Only sample once if there is a low number of points, and use replacement
                    subsample_to_TFrecord(writer, current_data, sample_size, 1, rng, replacement=True)
                else:
                    pass
                
                i += 1

                current_x += stride

            current_y += stride
            current_x = start_x

        print("Done!")
    return

def split_validation_data(full_data):
    # Take 25% of the training data (1/4 of the circular data)
    min_x = -20
    min_y = 0
    max_x = 0
    max_y = 20
    
    # Gather validation data
    mask =  (full_data[:,0] >= min_x) & (full_data[:,0] < max_x) & (full_data[:,1] >= min_y) & (full_data[:,1] < max_y)
    validation_data = full_data[mask]
    training_data = full_data[~mask]

    return training_data, validation_data


if __name__ == '__main__':
    # Min data = -20,-20, 0
    # Max data = 20, 20, ~55

    # Load the data without labels and other separated
    full_data = load_dataset(split = False)

    # Split training and validation data
    training_data, validation_data = split_validation_data(full_data)
  
    # Length of data
    print("Full data length: {}".format(len(full_data)))
    print("training_data length: {}".format(len(training_data)))
    print("validation_data length: {}".format(len(validation_data)))
    
    # Roughly an 80/20 training data split (22.7% validation data)

    # Set window width for current experiment
    window_width = 3
    print("Window width for current experiment: {}m".format(window_width))
    
    print("Packaging training data...")
    sliding_window(file_location="data/training_data.tfrecord", full_data=training_data, window_width=window_width, sample_num=20)

    print("Packaging validation data...")
    sliding_window(file_location="data/validation_data.tfrecord", full_data=validation_data, window_width=window_width, sample_num=1)

