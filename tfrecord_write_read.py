# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = Image.open(filename)
    if image.mode != 'RGB':
        image=image.convert('RGB')
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes() # convert image to raw data bytes in the array.

#  base_path + img_file_list with label
def write_tfrecord(img_filelist_path, base_path, tfrecord_file):
    # load image
    num_samples = 0
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        # write label, shape, and image content to the TFRecord file
        with open(img_filelist_path, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                img_path = os.path.join(base_path, img_name)
                shape, binary_image = get_image_binary(img_path)
                feature = tf.train.Features(feature={
                        'label': _int64_feature(int(label)),
                        'shape': _bytes_feature(shape),
                        'image': _bytes_feature(binary_image)
                        })
                example = tf.train.Example(features=feature)
                writer.write(example.SerializeToString())
                num_samples+=1
    print("Number of samples: {}".format(num_samples))

# read in tfrecord
def read_tfrecord(tfrecord_file):
    tfrecord_file_queue = tf.train.string_input_producer([tfrecord_file], name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'shape': tf.FixedLenFeature([], tf.string),
               'image': tf.FixedLenFeature([], tf.string)
               }
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features=feature, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.cast(image, tf.float32)*1.0/255
    image = tf.reshape(image, shape)
    label = tf.cast(tfrecord_features['label'], tf.int32)

    return label, image, shape

def get_batch(tfrecord_file, batch_size, num_threads=4, shuffle=False, min_after_dequeue=None):
    label, image, shape = read_tfrecord(['E:/myrecord.tfrecord'])

    if min_after_dequeue is None:
        min_after_dequeue = batch_size * 10
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                    capacity=capacity,num_threads=num_threads,
                                                    min_after_dequeue=min_after_dequeue)
    else:
        img_batch, label_batch = tf.train.batch([image, label], batch_size,
                                                capacity=capacity, num_threads=num_threads,
                                                allow_smaller_final_batch=True)


def main():
    # assume the image has the label Chihuahua, which corresponds to class number 1

    # test for write tfrecord demo
    write_tfrecord("E:/filelist.txt", "E:/mytest", 'E:/myrecord.tfrecord')

    # test for read in tfrecord
    label, image, shape = read_tfrecord('E:/myrecord.tfrecord')
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        label, image, shape = sess.run([label, image, shape])
        coord.request_stop()
        coord.join(threads)
    print(label)
    print(shape)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()
