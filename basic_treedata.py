import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
from pnet2_layers.layers import Pointnet_SA
import numpy as np
import pickle
from models.sem_seg_model import SEM_SEG_Model

  
def load_dataset():
    with open("data/plot_annotations.p", "rb") as f:
        annotations = pickle.load(f)
    data = np.asarray(annotations)
    features = data[:, 0:3] # x, y, z
    labels = data[:, 3] # label
    return features, labels

def generate_dataset(features, labels, batch_size):

    shuffle_buffer = 1000

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # x = features[:, 0]
    # y = features[:, 1]
    # z = features[:, 2]

    # dataset = tf.data.Dataset.from_tensor_slices((x,y,z,labels))

    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train(training_dataset: tf.data.Dataset):

    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
    
    callbacks = [
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
    ]

    model.build((config['batch_size'], 8192, 3))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(
        training_dataset,
        # validation_data=val_ds,
        # validation_steps=10,
        # validation_freq=1,
        callbacks=callbacks,
        epochs=1,
        verbose=1
    )

if __name__ == '__main__':

    # Set it up such that only 90% of the GPU memory is used, prevents crashing
    tempconfig = tf.compat.v1.ConfigProto()
    tempconfig.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
    tempconfig.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=tempconfig))

    # Parameters for the model and training
    config = {
        # 'train_ds' : 'data/scannet_train.tfrecord',
        # 'val_ds' : 'data/scannet_val.tfrecord',
        'log_dir' : 'trees_1',
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 4,
        'num_classes' : 21,
        'lr' : 0.001,
        'bn' : False,
    }

    features, labels = load_dataset()
    
    # Generate a Dataset object from tensorflow using our features and labels
    dataset = generate_dataset(features, labels, config['batch_size'])

    train(dataset)