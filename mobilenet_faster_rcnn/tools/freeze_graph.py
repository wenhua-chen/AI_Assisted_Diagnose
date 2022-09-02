#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import tensorflow as tf
import numpy as np
import os, cv2
import argparse
import pprint

from nets.vgg16 import vgg16
from nets.mobilenet_v1 import mobilenetv1
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v2 import mobilenetv2

from model.config import cfg, cfg_from_file, cfg_from_list


CLASSES = ('__background__', 'face','face1','face2')

NETS = ["vgg16", "res101", "res50", "mobile"]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument("-n", '--net', dest='net', help='Network to use [vgg16 res101]',
                        choices=NETS, default='res101')
    parser.add_argument("-m", "--model", dest="ckpt_path", type=str)
    parser.add_argument('-v', '--version', dest='version',
                      help='1 2',
                      default=1, type=int)
    parser.add_argument('-c', '--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('-cls', '--cls_n', dest='cls_n',
                        help='cls_num', default=3, type=int)
    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg['TEST'])

    # model path
    demonet = args.net
    ckpt_path = args.ckpt_path
    print("args: ", demonet, ckpt_path)
    if not ckpt_path.endswith(".ckpt"):
        print(ckpt_path)
        print("ckpt path error. Not exists or not a tensorflow checkpoint file")
        return
    freeze_graph_name = ckpt_path.replace(".ckpt", "_freeze.pb")

    print(ckpt_path + '.meta')
    if not os.path.isfile(ckpt_path + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(ckpt_path + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        if args.version == 1:
            net = mobilenetv1()
        else:
            net = mobilenetv2()
    else:
        raise NotImplementedError
    net.create_architecture("TEST", args.cls_n,
                          tag='default',
                          anchor_scales=[4,8,16,32],
                          anchor_ratios=cfg.ANCHOR_RATIOS)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    print('Loaded network {:s}'.format(ckpt_path))


    output_names = [out.op.name for out in net._predictions.values()]
    for k, v in net._predictions.items():
        print(k, v.name, sep="\t\t")
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = sess.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference([]))
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, output_names, freeze_var_names)
        tf.train.write_graph(frozen_graph, ".", freeze_graph_name, as_text=False)
        print('freeze_graph_name',freeze_graph_name)
if __name__ == '__main__':
    main()
