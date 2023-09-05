import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
from pnet2_layers.layers import Pointnet_SA
from models.sem_seg_model import SEM_SEG_Model
import os

def countRecords(ds:tf.data.Dataset):
  count = 0

  if tf.executing_eagerly():
    # TF v2 or v1 in eager mode
    for r in ds:
      count = count+1
  else:  
    # TF v1 in non-eager mode
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    next_batch = iterator.get_next()
    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(next_batch)
          count = count+1    
      except tf.errors.OutOfRangeError:
        pass
  
  return count

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

        points = tf.reshape(points, (n_points, 3))
        labels = tf.reshape(labels, (n_points, 1))

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


def train():

    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

    # Retreive dataset from TFRecord file    
    assert os.path.isfile(config['train_ds']), '[Error] Dataset path not found'
    dataset = tf.data.TFRecordDataset(config['train_ds'])

    # Split dataset into 80% training and 20% validation
    train_length = int(0.8 * countRecords(dataset))
    dataset_train = dataset.take(train_length)
    dataset_val = dataset.skip(train_length)

    # Prepare datasets
    dataset_train = prepare_data(dataset_train, config['batch_size'])
    dataset_val = prepare_data(dataset_val, config['batch_size'])

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
        metrics=[keras.metrics.SparseCategoricalAccuracy(),
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

    hist = model.fit(
        dataset_train,
        validation_data=dataset_val,
        # validation_steps= train_length // config['batch_size'],
        # validation_freq=1,
        callbacks=callbacks,
        epochs=10,
        verbose=1,
        # steps_per_epoch=train_length // config['batch_size']
    )

if __name__ == '__main__':

    # Set it up such that only 90% of the GPU memory is used, prevents crashing
    tempconfig = tf.compat.v1.ConfigProto()
    tempconfig.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
    tempconfig.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=tempconfig))

    # Parameters for the model and training
    config = {
         'train_ds' : 'data/plot_annotations_training.tfrecord', # 97190 examples, 20% is 19438
        # 'val_ds' : 'data/plot_annotations_validation.tfrecord',
        'log_dir' : 'trees_full',
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 20,
        'num_classes' : 4,
        'lr' : 0.001,
        'bn' : False,
    }

    print("____________________________________________")
    print("Have you checked you are using the correct number of points and the correct tfrecord file?")
    print("____________________________________________")


    train()