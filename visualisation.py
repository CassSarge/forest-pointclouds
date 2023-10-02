import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras.backend import set_session
from pnet2_layers.layers import Pointnet_SA
from models.sem_seg_model import SEM_SEG_Model
import os
from basic_treedata import prepare_data
import matplotlib.pyplot as plt
import numpy as np
import re

def create_model():
    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])
    
    model.build((config['batch_size'], 8192, 3))

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

    model.summary()

    return model


# main function for visualisation of model training
if __name__ == '__main__':
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
        'log_dir' : 'trees_{}'.format(1.0),
        'log_freq' : 10,
        'test_freq' : 100,
        'batch_size' : 1,
        'num_classes' : 4,
        'lr' : 0.001,
        'bn' : True,
    }

    # Prepare data
    dataset_train= tf.data.TFRecordDataset(config['train_ds'])
    dataset_val= tf.data.TFRecordDataset(config['val_ds'])

    # Prepare datasets
    dataset_train = prepare_data(dataset_train, config['batch_size'])
    dataset_val = prepare_data(dataset_val, config['batch_size'])

    # Create model
    model = create_model()

   #  # Test the model while untrained
   #  print("Testing untrained model...")
   #  history = model.evaluate(dataset_val, verbose=1)
   #  print("Untrained model, accuracy: {:5.2f}%".format(100 * history[1]))
    
    # Load weights
    checkpoint_dir = './logs/{}/model/weights'.format(config['log_dir'])
    model.load_weights(checkpoint_dir).expect_partial()
    

   #  # Test the model while trained
   #  print("Testing trained model...")
   #  history = model.evaluate(dataset_val, verbose=1)
   #  print("Trained model, accuracy: {:5.2f}%".format(100 * history[1]))

    # Make a prediction
    batch_size = 1
    num_examples = 42
    num_points = 8192

    # Predict
    # dataset = dataset_val.take(1)
    for points, labels in dataset_val:
        probabilities = model.predict(points)
        break

    predicted_labels = np.argmax(probabilities, axis=-1)
    
    predicted_labels = predicted_labels.reshape((num_points))
    points = points.numpy().reshape((num_points, 3))


    class_labels = {
        0: 'Foliage',
        1: 'Stem',
        2: 'Ground',
        3: 'Undergrowth'
    }

    # Convert original labels to integers
    true_labels = labels.numpy().reshape((num_points))
    true_labels = (np.rint(true_labels)).astype(int)

    # Plot the point cloud with the predicted labels
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # First subplot

    ax = fig.add_subplot(1,2,1, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=predicted_labels, cmap='inferno')
    ax.set_title('Predicted labels')

    # Create a custom legend
    handles, labels = scatter.legend_elements()
    custom_labels = [class_labels[int(re.findall(r'\d+', label)[0])] for label in labels]
    legend1 = ax.legend(handles, custom_labels, loc="lower left", title="Classes")
    ax.add_artist(legend1)

    # Second subplot

    ax = fig.add_subplot(1,2,2, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=true_labels, cmap='inferno')
    ax.set_title('True labels')

    # Create a custom legend
    handles, labels = scatter.legend_elements()
    custom_labels = [class_labels[int(re.findall(r'\d+', label)[0])] for label in labels]
    legend1 = ax.legend(handles, custom_labels, loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()