import os
import pickle
import numpy as np

def shorten_ply_file(filename: str, new_filename: str, n_points: int):
    # Open a .ply file and randomly remove 80% of points
    with open(filename, "rb") as f:
        ply_data = f.readlines()

    # save the first 10 lines
    header = ply_data[:10]
    print(header)

    # Delete data except for 8192 random points
    ply_data = ply_data[10:]
    ply_data = np.asarray(ply_data)
    length = len(ply_data)
    print("Length of ply data: {}".format(length))
    indices = np.arange(length)
    np.random.shuffle(indices)
    indices = indices[:n_points]
    ply_data = ply_data[indices]

    print("Length of ply data after shuffling: {}".format(len(ply_data)))

    # Save the remaining points to a new .ply file
    with open(new_filename, "wb") as f:
        f.writelines(header)
        f.writelines(ply_data)

if __name__ == '__main__':

    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    print(data.shape) # (10829404, 4) (x, y, z, label)
    print(data[1,:])
    print(type(data)) # numpy.ndarray
    print(data[1:200, :]) # (10829404, 4) (x, y, z, label

    features = data[:, 0:3] # Features between 
    labels = data[:, 3] 
    print(features.shape) # (10829404, 3)
    print(labels.shape) # (10829404,)
    print(features[1:50, :3]) #
    print(labels[1:50])

    print(type(features))
    print(type(features[1]))
    print(type(features[1][1]))

    print(type(labels))
    print(type(labels[1]))

