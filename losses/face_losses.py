import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.python.keras.layers import Layer
from config import config as cfg


def make_logits(embedding, label_one_hot, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
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
                label_one_hot = label_one_hot * s_m
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
                diff = body * s - fc7
                body = tf.multiply(label_one_hot, diff)
                fc7 = fc7 + body
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


def make_logits_v2(embedding, one_hot_label, class_num, loss_type='margin_softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False):
    embedding_size = embedding.get_shape().as_list()[-1]
    if w is None:
        w = tf.Variable(tf.random_normal([embedding_size, class_num], stddev=0.01), name='fc7_weight')
    if loss_type == 'margin_softmax':
        embedding_norm = tf.norm(embedding, axis=-1, keepdims=True, name='fc1n')
        embedding = embedding / embedding_norm
        w_norm = tf.norm(w, axis=0, keepdims=True)
        w = w / w_norm
        embedding_norm_scale = embedding * s
        fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
        if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
            if m1 == 1.0 and m2 == 0.0:
                s_m = s * m3
                label_one_hot = one_hot_label * s_m
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
                mask = 1 - one_hot_label
                fc7 = fc7*mask + body*one_hot_label
    else:
        fc7 = tf.matmul(embedding, w)
        if use_bias:
            bias = tf.Variable(tf.zeros([class_num, ]), name='fc7_bias')
            fc7 = tf.add(fc7, bias)
    return fc7


class FaceCategoryOutput(Layer):
    def __init__(self, units, loss_type='margin_softmax', act='softmax', s=64.0, m1=1.0, m2=0.5, m3=0.0, w=None, use_bias=False, name='face_category'):
        super(FaceCategoryOutput, self).__init__(name=name)
        self.units = units
        self.loss_type = loss_type
        self.act = act
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.use_bias = use_bias
        self.w = w

    def build(self, input_shape):
        if self.w is None:
            self.w = self.add_weight(name='fc7_weight',
                                     shape=(input_shape[-1], self.units),
                                     initializer=tf.random_normal_initializer(stddev=0.01),
                                     trainable=True)
        if self.loss_type != 'margin_softmax' and self.use_bias:
            self.b = self.add_weight(name='fc7_bias',
                                     shape=[self.units, ],
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)

    def call(self, inputs, label):
        label_one_hot = tf.one_hot(label, self.units)
        if self.loss_type == 'margin_softmax':
            embedding_norm = tf.norm(inputs, axis=-1, keepdims=True, name='fc1n')
            embedding = inputs / embedding_norm
            w_norm = tf.norm(self.w, axis=0, keepdims=True)
            w = self.w / w_norm
            embedding_norm_scale = embedding * self.s
            fc7 = tf.matmul(embedding_norm_scale, w, name='fc7')
            if self.m1 != 1.0 or self.m2 != 0.0 or self.m3 != 0.0:
                if self.m1 == 1.0 and self.m2 == 0.0:
                    s_m = self.s * self.m3
                    label_one_hot = label_one_hot * s_m
                    fc7 = fc7 - label_one_hot
                else:
                    cos_t = fc7 / self.s
                    t = tf.math.acos(cos_t)
                    if self.m1 != 1.0:
                        t = t * self.m1
                    if self.m2 > 0.0:
                        t = t + self.m2
                    body = tf.math.cos(t)
                    if self.m3 > 0.0:
                        body = body - self.m3
                    diff = body * self.s - fc7
                    body = tf.multiply(label_one_hot, diff)
                    fc7 = fc7 + body
        else:
            fc7 = tf.matmul(inputs, self.w)
            if self.use_bias:
                fc7 = tf.add(fc7, self.b)
        if self.act:
            fc7 = keras.layers.Activation(self.act)(fc7)
        return fc7

    def get_config(self):
        return {'units': self.units,
                'act': self.act,
                'loss_type': self.loss_type,
                's': self.s,
                'm1': self.m1,
                'm2': self.m2,
                'm3': self.m3,
                'use_bias': self.use_bias}


if __name__ == '__main__':
    cfg.debug = True
    tf.enable_eager_execution()
    batch_num = 1
    batch_size = 2048
    feature_dim = 512
    persons = 100
    embeddings = tf.constant(tf.random_normal([batch_size, feature_dim]))
    ws = tf.Variable(tf.random_normal([feature_dim, persons]))
    labels = tf.constant(tf.random_uniform([batch_size, ], maxval=persons, dtype=tf.int32))
    labels_one_hot = tf.one_hot(labels, persons)

    tmp = make_logits(embeddings, labels_one_hot, persons, w=ws)
    face_out = FaceCategoryOutput(persons, w=ws, act=None)

    start = time.time()
    for _ in range(batch_num):
        out1 = make_logits(embeddings, labels_one_hot, persons, w=ws)
    t1 = time.time()
    print('logits 1 cost:', t1-start, 's')
    print('out1:', out1)
    for _ in range(batch_num):
        out2 = make_logits_v2(embeddings, labels_one_hot, persons, w=ws)
    t2 = time.time()
    print('logits 2 cost:', t2 - t1, 's')
    print('out2:', out2)
    for _ in range(batch_num):
        out3 = face_out(embeddings, labels)
    t3 = time.time()
    print('logits 3 cost:', t3 - t2, 's')
    print('out3:', out3)
    print('out1-out2', tf.reduce_sum(out1 - out2))
    print('out3-out2', tf.reduce_sum(out3 - out2))
    print('out3-out1', tf.reduce_sum(out3 - out1))

