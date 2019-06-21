from tensorflow import keras
from tensorflow.python.keras.layers import Layer
from losses import face_losses

import tensorflow as tf


def activation(act_type, name='act'):
    if act_type is None:
        act = None
    elif act_type == 'prelu':
        act = keras.layers.PReLU(name=name)
    else:
        act = keras.layers.Activation(act_type, name=name)
    return act


def get_fc1(last_conv, training, embedding_size, fc_type, bn_mom=0.9, wd=0.0005):
    body = last_conv
    if fc_type == 'Z':
        body = keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn1')(body, training=training)
        body = keras.layers.Dropout(rate=0.4)(body, training=training)
        fc1 = body
    elif fc_type == 'E':
        body = keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn1')(body, training=training)
        body = keras.layers.Dropout(rate=0.4)(body, training=training)
        fc1 = keras.layers.Dense(units=embedding_size, name='pre_fc1')(body)
        fc1 = keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='fc1')(fc1, training=training)
    elif fc_type == 'FC':
        body = keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn1')(body, training=training)
        fc1 = keras.layers.Dense(units=embedding_size, name='pre_fc1')(body)
        fc1 = keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='fc1')(fc1, training=training)
    elif fc_type == "GDC":  # mobilefacenet_v1
        conv_6_dw = DWConvBnAct(kernel_size=7, stride=1, padding=0, act_type=None, wd=wd, bn_mom=bn_mom,
                                name='conv_6dw7_7')(last_conv, training)
        conv_6_f = keras.layers.Dense(units=embedding_size, name='pre_fc1')(conv_6_dw, training)
        fc1 = keras.layers.BatchNormalization(eps=2e-5, momentum=bn_mom, name='fc1')(conv_6_f, training)
    return fc1


class ImageStandardization(Layer):
    def __init__(self, mean=127.5, std_inv=0.0078125):
        super(ImageStandardization, self).__init__()
        self.mean = mean
        self.std_inv = std_inv

    def call(self, inputs):
        image = inputs - self.mean
        image = image * self.std_inv
        return image


class ConvBnAct(Layer):
    def __init__(self, filters=1, kernel_size=3, stride=1, padding=1, act_type='prelu',
                 use_bias=False, wd=0.0005, bn_mom=0.9, name='conv_bn_act'):
        super(ConvBnAct, self).__init__(name=name)

        self.pad = keras.layers.ZeroPadding2D(padding=padding)
        self.conv = keras.layers.Conv2D(filters, kernel_size, stride,
                                        use_bias=use_bias,
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=keras.regularizers.l2(wd))
        self.bn = keras.layers.BatchNormalization(momentum=bn_mom)
        self.act = activation(act_type)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class DWConvBnAct(Layer):
    def __init__(self, kernel_size=3, stride=1, padding=1, act_type='prelu',
                 use_bias=False, wd=0.0005, bn_mom=0.9, name='dw_conv_bn_act'):
        super(DWConvBnAct, self).__init__(name=name)

        self.pad = keras.layers.ZeroPadding2D(padding=padding)
        self.dw_conv = keras.layers.DepthwiseConv2D(kernel_size, stride,
                                                    use_bias=use_bias,
                                                    depthwise_initializer='glorot_normal',
                                                    depthwise_regularizer=keras.regularizers.l2(wd))
        self.bn = keras.layers.BatchNormalization(momentum=bn_mom)
        self.act = activation(act_type)

    def call(self, inputs, training=None):
        x = self.dw_conv(inputs)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class DWBlock(Layer):
    def __init__(self, num_out, kernel_size=3, stride=1, padding=1, num_group=1, act_type='prelu',
                 use_bias=False, wd=0.0005, bn_mom=0.9, name='dw_block', suffix=''):
        super(DWBlock, self).__init__(name=name+suffix)

        self.expand = ConvBnAct(filters=num_group, kernel_size=1, stride=1, padding=0, act_type=act_type,
                                use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='expand')
        self.depth_wise = DWConvBnAct(kernel_size=kernel_size, stride=stride, padding=padding, act_type=act_type,
                                      use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='depth_wise')
        self.project = ConvBnAct(filters=num_out, kernel_size=1, stride=1, padding=0, act_type=None,
                                 use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='project')

    def call(self, inputs, training=None):
        x = self.expand(inputs, training=training)
        x = self.depth_wise(x, training=training)
        x = self.project(x, training=training)
        return x


class FaceCategoryOutput(Layer):
    def __init__(self, units, loss_type='margin_softmax', act=None, s=64.0, m1=1.0, m2=0.5, m3=0.0, use_bias=False, name='face_category'):
        super(FaceCategoryOutput, self).__init__(name=name)
        self.units = units
        self.loss_type = loss_type
        self.act = act
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.use_bias = use_bias

    def build(self, input_shape):
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

