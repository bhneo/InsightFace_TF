import tensorflow as tf
import time
from config import config as cfg


def make_logits(embedding, label, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding/embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w/w_norm
        embedding_norm_scale = embedding * s
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = tf.one_hot(label, depth=class_num, on_value=s_m, off_value=0.0)
                fc7 = fc7 - label_one_hot
            else:
                cos_t = fc7 / s
                t = tf.math.acos(cos_t)
                if m1 != 1.0:
                    t = t * m1
                if m2 > 0.0:
                    t = t + m2
                body = tf.math.cos(t)
                if m3 > 0.0:
                    body = body - m3
                diff = body * s - cos_t
                label_one_hot = tf.one_hot(label, depth=class_num, on_value=1.0, off_value=0.0)
                body = tf.multiply(label_one_hot, diff)
                fc7 = fc7 + body
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def make_logits_v2(embedding, label, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding / embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w / w_norm
        embedding_norm_scale = embedding * s
        # if cfg.debug:
        #     test_norm = tf.norm(embedding, axis=-1)
        #     print('embedding norm test:', test_norm)
        #     test_norm = tf.norm(w, axis=0)
        #     print('weights norm test:', test_norm)
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = tf.one_hot(label, depth=class_num, on_value=s_m, off_value=0.0)
                fc7 = fc7 - label_one_hot
            else:
                cos_t = fc7 / s
                t = tf.math.acos(cos_t)
                if m1 != 1.0:
                    t = t * m1
                if m2 > 0.0:
                    t = t + m2
                body = tf.math.cos(t)
                if m3 > 0.0:
                    body = body - m3
                body = body * s
                label_one_hot = tf.one_hot(label, depth=class_num, on_value=1.0, off_value=0.0)
                mask = 1 - label_one_hot
                fc7 = fc7*mask + body*label_one_hot
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


if __name__ == '__main__':
    cfg.debug = True
    tf.enable_eager_execution()
    batch_num = 5000
    batch_size = 2048
    feature_dim = 512
    persons = 100
    embeddings = tf.constant(tf.random_normal([batch_size, feature_dim]))
    ws = tf.Variable(tf.random_normal([feature_dim, persons]))
    labels = tf.constant(tf.random_uniform([batch_size, ], maxval=persons, dtype=tf.int32))

    tmp = make_logits(embeddings, labels, persons, w=ws)

    start = time.time()
    for _ in range(batch_num):
        out1 = make_logits(embeddings, labels, persons, w=ws)
    t1 = time.time()
    print('logits 1 cost:', t1-start, 's')
    print('out1:', out1)
    for _ in range(batch_num):
        out2 = make_logits_v2(embeddings, labels, persons, w=ws)
    t2 = time.time()
    print('logits 2 cost:', t2 - t1, 's')
    print('out2:', out2)

