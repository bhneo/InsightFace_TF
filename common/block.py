import tensorflow as tf
import mxnet as mx
from tensorflow import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model
from config import config


class ImageStandardization(Layer):
    def __init__(self, mean=127.5, std_inv=0.0078125):
        super(ImageStandardization, self).__init__()
        self.mean = mean
        self.std_inv = std_inv

    def call(self, inputs, **kwargs):
        image = inputs - self.mean
        image = image * self.std_inv
        return image


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


class ConvBnAct(Model):
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

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class DWConvBnAct(Model):
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

    def call(self, input_tensor, training=False, **kwargs):
        x = self.dw_conv(input_tensor)
        x = self.bn(x, training=training)
        if self.act:
            x = self.act(x)
        return x


class DWBlock(Model):
    def __init__(self, num_out, kernel_size=3, stride=1, padding=1, num_group=1, act_type='prelu',
                 use_bias=False, wd=0.0005, bn_mom=0.9, name='dw_block', suffix=''):
        super(DWBlock, self).__init__(name=name+suffix)

        self.expand = ConvBnAct(filters=num_group, kernel_size=1, stride=1, padding=0, act_type=act_type,
                                use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='expand')
        self.depth_wise = DWConvBnAct(kernel_size=kernel_size, stride=stride, padding=padding, act_type=act_type,
                                      use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='depth_wise')
        self.project = ConvBnAct(filters=num_out, kernel_size=1, stride=1, padding=0, act_type=None,
                                 use_bias=use_bias, wd=wd, bn_mom=bn_mom, name='project')

    def call(self, input_tensor, training=False, **kwargs):
        x = self.expand(input_tensor, training=training)
        x = self.depth_wise(x, training=training)
        x = self.project(x, training=training)
        return x


