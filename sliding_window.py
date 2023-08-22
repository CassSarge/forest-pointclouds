import numpy as np
import pickle
import tensorflow as tf

from generateTFRecord import load_dataset, basic_example_from_data

def subsample_to_TFrecord(writer, data, sample_size, n_subsamples, randomizer):

    # Repeatedly subsample the data and save it to the TFRecord file
    for i in range(n_subsamples):
        subsampled_data = randomizer.choice(data, sample_size, replace=True)
        current_example = basic_example_from_data(subsampled_data[:, 0:3], subsampled_data[:, 3])
        writer.write(current_example.SerializeToString())


    return

def sliding_window(full_data, window_width: int, stride: int, sample_size: int = 8192):

    start_x = -19
    start_y = -19

    end_x = 19
    end_y = 19

    current_x = start_x
    current_y = start_y

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
                    subsample_to_TFrecord(writer, current_data, sample_size, 30, rng)
                # Else if there are less than 8192 points, but more than 4096, we'll upsample
                elif num_points > sample_size / 2:
                    # Only sample a bit if theres a low number of points
                    subsample_to_TFrecord(writer, current_data, sample_size, 5, rng)
                else:
                    pass
                
                print("i: {}, num_points: {}, Current x: {}, Current y: {}".format(i, num_points, current_x, current_y))

                
                i += 1

                current_x += stride

            current_y += stride
            current_x = start_x

        print("Done!")
    return



if __name__ == '__main__':
    # Min data = -20,-20, 0
    # Max data = 20, 20, ~55
    full_data = load_dataset(False)
  
    sliding_window(full_data, 1, 0.5)

