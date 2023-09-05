import numpy as np
import pickle
import tensorflow as tf

from generateTFRecord import load_dataset, basic_example_from_data

def subsample_to_TFrecord(writer, data, sample_size, n_subsamples, randomizer, replacement = False):

    # Repeatedly subsample the data and save it to the TFRecord file
    for i in range(n_subsamples):
        subsampled_data = randomizer.choice(data, sample_size, replace=replacement)
        current_example = basic_example_from_data(subsampled_data[:, 0:3], subsampled_data[:, 3])
        writer.write(current_example.SerializeToString())


    return

def sliding_window(full_data, window_width: int, overlap: float= 0.3, sample_size: int = 8192):

    start_x = -20
    start_y = -20

    end_x = 20
    end_y = 20

    current_x = start_x
    current_y = start_y
    
    stride = int((1-overlap)*window_width)

    i = 0

    rng = np.random.default_rng(12345)
    with tf.io.TFRecordWriter("data/full_w_subsampling.tfrecord") as writer:

        while current_y < end_y:

            while current_x < end_x:
                # Get points within window
                current_data = full_data[(full_data[:,0] >= current_x - window_width) & (full_data[:,0] < current_x + window_width) & (full_data[:,1] >= current_y - window_width) & (full_data[:,1] < current_y + window_width)]
                # Get labels within window
                # Count number of points and print
                (num_points, _) = current_data.shape
                # (num_labels) = current_labels.shape
                # Point densities range from about 0 to 40000 within a window right now
                # Choose a random subsample of these
                if num_points > sample_size:
                    subsample_to_TFrecord(writer, current_data, sample_size, 20, rng)
                # Else if there are less than 8192 points, but more than 4096, we'll upsample
                elif num_points > sample_size / 2:
                    # Only sample once if there is a low number of points, and use replacement
                    subsample_to_TFrecord(writer, current_data, sample_size, 1, rng, replacement=True)
                else:
                    pass
                
                print("i: {}, num_points: {}, Current x: {}, Current y: {}".format(i, num_points, current_x, current_y))

                
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

    print("{}".format(type((full_data[:,0] >= -20) & (full_data[:,0] < 0) & (full_data[:,1] >= 0) & (full_data[:,1] < 20))))

    training_data, validation_data = split_validation_data(full_data)
  
    # Length of full data
    print("Full data length: {}".format(len(full_data)))
    print("training_data length: {}".format(len(training_data)))
    print("validation_data length: {}".format(len(validation_data)))
    
    # Roughly an 80/20 training data split (22.7% validation data)

    # sliding_window(full_data, 1, 0.5)

