import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt



def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    # r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    # img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def get_training_data(tf_record_path, batch_size=128):
    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element


def view_training_data():
    sess = tf.Session()
    # training dataset api config
    tf_records = os.path.join('data', 'train.tfrecords')
    if not os.path.exists(tf_records):
        raise FileExistsError(tf_records)
    iterator, next_element = get_training_data(tf_records)
    # begin iteration
    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, labels = sess.run(next_element)
                plt.imshow(images[1, ...])
                plt.show()
            except tf.errors.OutOfRangeError:
                print("End of dataset")


def load_eval_data(dataset_list):
    ver_list = []
    ver_name_list = []
    for db in dataset_list:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)


def load_bin(db_name, image_size, args):
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    data_list = []
    for _ in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0,1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


if __name__ == '__main__':
    view_training_data()
