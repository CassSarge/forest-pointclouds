import tensorflow as tf

shuffle_buffer = 1000
n_points = 8192


def _extract_fn(data_record):

		in_features = {
			'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
			'labels': tf.io.FixedLenFeature([n_points], tf.int64)
		}

		return tf.io.parse_single_example(data_record, in_features)


raw_dataset = tf.data.TFRecordDataset("data/scannet_train.tfrecord")

print(raw_dataset)
for raw_record in raw_dataset.take(1):
   print(repr(raw_record))

