import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
from pnet2_layers.layers import Pointnet_SA
from models.sem_seg_model import SEM_SEG_Model
import os
import matplotlib.pyplot as plt
import pickle

def plot_result(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def prepare_data(dataset, batch_size):

    n_points = 8192

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

        points = tf.reshape(points, shape=(n_points, 3))
        labels = tf.reshape(labels, shape=(n_points, 1))

        shuffle_idx = tf.range(points.shape[0])
        shuffle_idx = tf.random.shuffle(shuffle_idx)
        points = tf.gather(points, shuffle_idx)
        labels = tf.gather(labels, shuffle_idx)

        return points, labels

    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train(config: dict):

    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

    # Retreive dataset from TFRecord file    
    assert os.path.isfile(config['train_ds']), '[Error] Dataset path not found'
    
    dataset_train= tf.data.TFRecordDataset(config['train_ds'])
    dataset_val= tf.data.TFRecordDataset(config['val_ds'])

    # Prepare datasets
    dataset_train = prepare_data(dataset_train, config['batch_size'])
    dataset_val = prepare_data(dataset_val, config['batch_size'])

    callbacks = [
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights'.format(config['log_dir']), monitor='sparse_cat_acc', save_best_only=True, save_weights_only=True)
    ]

    model.build((config['batch_size'], 8192, 3))

    # model.compute_output_shape(input_shape=(config['batch_size'], 8192, 3))

    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        run_eagerly=True,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="sparse_cat_acc"),
                 keras.metrics.IoU(
                    name="meanIoU",
                    num_classes=config['num_classes'],
                    target_class_ids = [0,1,2,3],
                    sparse_y_true=True, 
                    sparse_y_pred=False
                 ),
                 keras.metrics.IoU(
                    name="FoliageIoU",
                    num_classes=config['num_classes'],
                    target_class_ids = [0],
                    sparse_y_true=True, 
                    sparse_y_pred=False
                 ),
                 keras.metrics.IoU(
                    name="StemIoU",
                    num_classes=config['num_classes'],
                    target_class_ids = [1],
                    sparse_y_true=True, 
                    sparse_y_pred=False
                 ),
                 keras.metrics.IoU(
                    name="GroundIoU",
                    num_classes=config['num_classes'],
                    target_class_ids = [2],
                    sparse_y_true=True, 
                    sparse_y_pred=False
                 ),
                 keras.metrics.IoU(
                    name="UndergrowthIoU",
                    num_classes=config['num_classes'],
                    target_class_ids = [3],
                    sparse_y_true=True, 
                    sparse_y_pred=False
                 )]
    )

    train_length = dataset_train.reduce(0, lambda x, _: x + 1)
    val_length = dataset_val.reduce(0, lambda x, _: x + 1)

    print("Train length: {}".format(train_length))
    print("Val length: {}".format(val_length))

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        callbacks=callbacks,
        epochs=50,
        verbose=1,
    )

    plot_result(history, "loss")
    plot_result(history, "sparse_cat_acc")
    plot_result(history, "meanIoU")
    plot_result(history, "FoliageIoU")
    plot_result(history, "StemIoU")
    plot_result(history, "GroundIoU")
    plot_result(history, "UndergrowthIoU")

    # Save history
    with open('./logs/{}/trainHistoryDict'.format(config['log_dir']), 'wb') as f:
       pickle.dump(history.history, f)

def setup(window_width: int = 0):
    tf.compat.v1.enable_eager_execution()
    # Set it up such that only 90% of the GPU memory is used, prevents crashing
    tempconfig = tf.compat.v1.ConfigProto()
    tempconfig.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
    tempconfig.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=tempconfig))

    # Parameters for the model and training
    config = {
        'train_ds' : 'data/training_data.tfrecord', 
        'val_ds' : 'data/validation_data.tfrecord',
        'log_dir' : 'trees_{}'.format(window_width),
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 16,
        'num_classes' : 4,
        'lr' : 0.001,
        'bn' : True,
    }

    return config

if __name__ == '__main__':

    config = setup()
    train(config)

    print("____________________________________________")
    print("Have you checked you are using the correct number of points and the correct tfrecord file?")
    print("____________________________________________")
