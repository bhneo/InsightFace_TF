import os
import pickle
import numpy as np
import tensorflow as tf
import mxnet as mx
import matplotlib.pyplot as plt

from config import config


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def training_dataset(tf_record_path, batch_size=128, shuffle_buffer=50000):
    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer is not 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def get_training_pipeline(tf_record_path, batch_size=128, shuffle_buffer=50000):
    dataset = training_dataset(tf_record_path, batch_size, shuffle_buffer)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def count_training_data():
    sess = tf.Session()
    tf_records = os.path.join('data', 'train.tfrecords')
    if not os.path.exists(tf_records):
        raise FileExistsError(tf_records)
    batch_size = 128
    iterator, next_element = get_training_pipeline(tf_records, batch_size, 0)
    sess.run(iterator.initializer)
    dataset_size = 0
    steps = 0
    while True:
        try:
            images, labels = sess.run(next_element)
            dataset_size += images.shape[0]
            steps += 1
            if steps % 10000 == 0:
                print('steps', steps)
        except tf.errors.OutOfRangeError:
            print("Dataset size:", dataset_size)
            break


def view_training_data():
    sess = tf.Session()
    tf_records = os.path.join('data', 'train.tfrecords')
    if not os.path.exists(tf_records):
        raise FileExistsError(tf_records)
    iterator, next_element = get_training_pipeline(tf_records)
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                images /= 255.
                plt.figure()
                for k in range(16):
                    plt.subplot(4, 4, k+1)
                    plt.imshow(images[k, ...])
                    plt.text(0, 15, labels[k], fontdict={'color': 'red'})
                    plt.title(labels[k])
                plt.show()
            except tf.errors.OutOfRangeError:
                print("End of dataset")


def load_eval_data(dataset_list, image_size, path='data'):
    ver_list = []
    ver_name_list = []
    for db in dataset_list:
        print('begin db %s convert.' % db)
        data_set = load_bin(os.path.join(path, db+'.bin'), image_size)
        ver_list.append(data_set)
        ver_name_list.append(db)


def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, is_same_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, is_same_list = pickle.load(f, encoding='bytes')

    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(is_same_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(is_same_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img.asnumpy()
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, is_same_list


def load_valid_set(data_dir, dataset_list, image_size=(112, 112)):
    valid_list = []
    valid_name_list = []
    for name in dataset_list:
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = load_bin(path, image_size)
            valid_list.append(data_set)
            valid_name_list.append(name)
            print('valid set', name)
    return valid_list, valid_name_list


if __name__ == '__main__':
    # view_training_data()
    # count_training_data()
    load_valid_set('data', ['lfw', 'cfp_fp', 'agedb_30'])
