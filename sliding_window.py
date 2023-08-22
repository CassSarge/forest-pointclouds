import numpy as np
import pickle

from generateTFRecord import load_dataset

def sliding_window(features, labels, window_width: int, stride: int, sample_size: int = 8192):

    start_x = -19
    start_y = -19

    end_x = 19
    end_y = 19

    current_x = start_x
    current_y = start_y

    i = 0

    while current_y < end_y:

        while current_x < end_x:
            # Get points within window
            current_features = features[(features[:,0] >= current_x - window_width) & (features[:,0] < current_x + window_width) & (features[:,1] >= current_y - window_width) & (features[:,1] < current_y + window_width)]
            # Get labels within window
            # current_labels = labels[(features[:,0] >= current_x - window_width) & (features[:,0] < current_x + window_width) & (features[:,1] >= current_y - window_width) & (features[:,1] < current_y + window_width)]
            # Count number of points and print
            (num_points, _) = current_features.shape
            # (num_labels) = current_labels.shape
            # Point densities range from about 0 to 40000 within a window right now
            # This loop might suck investigate sorting methods?
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
    features, labels = load_dataset()
    print(features.shape)

    sliding_window(features, labels, 1, 0.5)

    # print(np.amin(features[:,0]))
    # print(np.amax(features[:,0]))
    # print(np.amin(features[:,1]))
    # print(np.amax(features[:,1]))
    # print(np.amin(features[:,2]))
    # print(np.amax(features[:,2]))

