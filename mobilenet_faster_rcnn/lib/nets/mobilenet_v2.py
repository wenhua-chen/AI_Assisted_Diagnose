# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import tensorflow.contrib.slim as slim

from model.config import cfg
from nets.mobilenet import mobilenet_v2
from nets.network import Network

def separable_conv2d_same(inputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D separable convolution with 'SAME' padding.
  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.
  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """

  # By passing filters=None
  # separable_conv2d produces only a depth-wise convolution layer
  if stride == 1:
    return slim.separable_conv2d(inputs, None, kernel_size,
                                  depth_multiplier=1, stride=1, rate=rate,
                                  padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.separable_conv2d(inputs, None, kernel_size,
                                  depth_multiplier=1, stride=stride, rate=rate,
                                  padding='VALID', scope=scope)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class mobilenetv2(Network):
    def __init__(self):
        super(mobilenetv2, self).__init__()

        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER_2
        self._scope = 'MobilenetV2'
        self._layer_to_split = 8

        assert (0 <= cfg.MOBILENET.FIXED_LAYERS_2 <= self._layer_to_split)
        self._head_fixed_def = copy.deepcopy(mobilenet_v2.V2_DEF)
        self._head_fixed_def["spec"] = self._head_fixed_def["spec"][:cfg.MOBILENET.FIXED_LAYERS_2]
        self._head_def = copy.deepcopy(mobilenet_v2.V2_DEF)
        self._head_def["spec"] = self._head_def["spec"][cfg.MOBILENET.FIXED_LAYERS_2:self._layer_to_split]
        self._tail_def = copy.deepcopy(mobilenet_v2.V2_DEF)
        self._tail_def["spec"] = self._tail_def["spec"][self._layer_to_split:]

    def _image_to_head(self, is_training, reuse=None):
        # Base bottleneck
        net_conv = self._image
        print("len of fixed head:", len(self._head_fixed_def["spec"]))
        print("len of head 2:", len(self._head_def["spec"]))
        if len(self._head_fixed_def["spec"]) > 0:
            with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                net_conv, _ = mobilenet_v2.mobilenet(net_conv, conv_defs=self._head_fixed_def,
                                                     depth_multiplier=self._depth_multiplier,
                                                     scope=self._scope, base_only=True, reuse=reuse)

        if len(self._head_def["spec"]) > 0:
            with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
                net_conv, _ = mobilenet_v2.mobilenet(net_conv, conv_defs=self._head_def,
                                                     depth_multiplier=self._depth_multiplier,
                                                     scope=self._scope, base_only=True, reuse=reuse)
        with tf.variable_scope(self._scope + "/head_part/seperate_1") as s:
            net_conv = separable_conv2d_same(net_conv, 3,
                                        stride=1,
                                        rate=1,
                                        scope=s)
            net_conv = slim.conv2d(net_conv, 512, [1, 1], stride=1)
        with tf.variable_scope(self._scope + "/head_part/seperate_2") as s:
            net_conv = separable_conv2d_same(net_conv, 3,
                                             stride=1,
                                             rate=1,
                                             scope=s)
            net_conv = slim.conv2d(net_conv, 512, [1, 1], stride=1)

        output = tf.identity(net_conv, name="image_to_head_output")
        self._act_summaries.append(output)
        self._layers['head'] = output

        return output

    def _head_to_tail(self, pool5, is_training, reuse=None):
        pool5 = slim.conv2d(pool5, 64, [1, 1], stride=1)
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
            net_conv, _ = mobilenet_v2.mobilenet(pool5, conv_defs=self._tail_def,
                                                 depth_multiplier=self._depth_multiplier,
                                                 scope=self._scope, base_only=True, reuse=reuse)
            # average pooling done by reduce_mean
            fc7 = tf.reduce_mean(net_conv, axis=[1, 2])
        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/Conv/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                # print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix MobileNet V2 layers..')
        print(" depth multiplier: ", self._depth_multiplier, _make_divisible(32 * self._depth_multiplier, 8))
        with tf.variable_scope('Fix_MobileNet_V2') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                Conv2d_0_rgb = tf.get_variable("Conv2d_0_rgb",
                                               [3, 3, 3, _make_divisible(32 * self._depth_multiplier, 8)],
                                               trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/Conv/weights": Conv2d_0_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv/weights:0"],
                                   tf.reverse(Conv2d_0_rgb / (255.0 / 2.0), [2])))
