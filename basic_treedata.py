import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
from pnet2_layers.layers import Pointnet_SA
import numpy as np
import pickle
from models.sem_seg_model import SEM_SEG_Model
import os


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[Error] Dataset path not found'

    # n_points = 8192
    # n_points = 10829404
    n_points = 100000

    shuffle_buffer = 1000
    
    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'labels': tf.io.FixedLenFeature([n_points], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        labels = sample['labels']

        points = tf.reshape(points, (n_points, 3))
        labels = tf.reshape(labels, (n_points, 1))

        shuffle_idx = tf.range(points.shape[0])
        shuffle_idx = tf.random.shuffle(shuffle_idx)
        points = tf.gather(points, shuffle_idx)
        labels = tf.gather(labels, shuffle_idx)

        return points, labels

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train():

    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
    
    training_dataset = load_dataset(config['train_ds'], config['batch_size'])
    validation_dataset = load_dataset(config['val_ds'], config['batch_size'])

    callbacks = [
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights'.format(config['log_dir']), monitor='sparse_categorical_accuracy', save_best_only=True)
    ]

    model.build((config['batch_size'], 8192, 3))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    hist = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        # validation_steps=10,
        # validation_freq=1,
        callbacks=callbacks,
        epochs=100,
        verbose=1,
        # steps_per_epoch=3
    )

if __name__ == '__main__':

    # Set it up such that only 90% of the GPU memory is used, prevents crashing
    tempconfig = tf.compat.v1.ConfigProto()
    tempconfig.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
    tempconfig.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=tempconfig))

    # Parameters for the model and training
    config = {
         'train_ds' : 'data/plot_annotations_training.tfrecord',
        'val_ds' : 'data/plot_annotations_validation.tfrecord',
        'log_dir' : 'trees_1',
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 10,
        'num_classes' : 4,
        'lr' : 0.001,
        'bn' : False,
    }

    print("____________________________________________")
    print("Have you checked you are using the correct number of points and the correct tfrecord file?")
    print("____________________________________________")

    train()