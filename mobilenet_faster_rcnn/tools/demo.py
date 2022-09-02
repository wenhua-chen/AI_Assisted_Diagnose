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
import sys
sys.path.append('/data/ubuntu/qiaoran/mobile_net_faster_rcnn/tools/')

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
import json

# CLASSES = ('__background__', 'wrinkle')
# CLASSES = ('__background__', 'bingzao')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_10000.ckpt',), 'res50': ('res50_faster_rcnn_iter_10000.ckpt',) , "mobile": ("mobile_faster_rcnn_iter_48000.ckpt")}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
"coco": ("coco_2014_train+coco_2014_valminusminival",),
"wrinkle": ("wrinkle_train",),
"ban": ("ban_new_train",),
"huangheban": ("huangheban_train",),
"wrinkle_old": ("wrinkle_train_old",),
"shangzonggebk_feimenzdzh": ("shangzonggebk_feimenzdzh_train",)}


os.environ["CUDA_VISIBLE_DEVICES"]="3"

def vis_gt(image, img_name,gt_json,out_dir):
    img_name_find = img_name.split('/')[-1]
    save_name = out_dir+img_name.split('.')[0] + '_gt.jpg'
    print('save_name', save_name)

    # json_path = os.path.join(data_folder, 'annotations/annotation.json')
    json_path = gt_json
    # json_path = '/data/ubuntu/qiaoran/mobile_net_faster_rcnn/data/CC_YanZhenQ_val/annotations/annotation.json'
    im = image[:, :, (2, 1, 0)].copy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    with open(json_path, 'r') as file:
        data = json.load(file)

    print('img_name_find', img_name_find)
    for item in data['images']:
        # print(img_name_find)
        if item['file_name'] == img_name_find:
          img_id = item['id']


    for item in data['annotations']:
        if item['image_id'] == img_id:
            bbox = item['bbox']
            cat_id = item['category_id']
            for i in data['categories']:
                if i['id'] == cat_id:
                    cat_name = i['name']
                    # print(cat_name)
                    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 204, 0), 2)
                    cv2.putText(im, '%s: ' % (cat_name), (bbox[0]+15, bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    2.0, (0, 255, 255), thickness=1)

    cv2.imwrite(save_name, im)
    return im

def vis_detections(image, class_name, dets, idx, thresh=0.3):
    """Draw detected bounding boxes."""
    # print(dets)
    inds = np.where(dets[:, -1] >= thresh)[0]
    # print(dets)
    if len(inds) == 0:
        print("found no target")
        return

    im = image[:, :, (2, 1, 0)].copy()
    print(im.shape)
    # cv2.imwrite('/home/lyc/tf-faster-rcnn/im_check.jpg', im)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print(bbox)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        im = cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,
            2, (255, 255, 255), 2, cv2.LINE_AA)

    # cv2.imwrite('/home/lyc/tf')
    save_file = "detections_" +str(idx)+class_name+".jpg"
    print(save_file)
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_file, im)
    # print ("save image to "+save_file)

    # im = image[:, :, (2, 1, 0)].copy()
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # for i in inds:
    #     bbox = dets[i, :4]
    #     score = dets[i, -1]
    #     bbox = list(map(int, bbox))
    #     bbox_file = str(idx)+"_"+str(i)+"_bbox.jpg"
    #     cv2.imwrite(bbox_file, im[bbox[1]:bbox[3], bbox[0]:bbox[2], :])

def vis_all_cls(image, image_name, output_dict):
    # print('image_name', image_name)
    save_name = image_name.split('.')[0] + '_pred.jpg'

    print('save_name', save_name)
    im = image[:, :, (2, 1, 0)].copy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for key in output_dict.keys():
        for item in output_dict[key]:
            bbox = item[:4]
            score = item[-1]
            im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            im = cv2.putText(im, '{:s} {:.3f}'.format(key, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(save_name, im)

def demo(sess, net, image_name,json_gt,out_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    # print('scores', scores)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    thresh = 0.5

    # import pickle
    # with open("final_pikle.pkl", "wb") as f:
    #     pickle.dump(boxes, f)

    output_bbox_dict = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # with open("dets.pkl", "wb") as f:
        #     pickle.dump(dets, f)
        keep = nms(dets, NMS_THRESH)

        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        dets = dets[inds, :]
        # dets = [dets[i, :] + [cls_ind] for i in range(dets.shape[0])]
        output_bbox_dict[cls] = dets
        # print('dets.shape', dets.shape)


        # print('dets', dets)
        # with open("dets_and_keep.pkl", "wb") as f:
        #     pickle.dump([dets, keep], f)

        # vis_detections(im, cls, dets, os.path.split(image_name)[-1].replace(".jpg", ""), thresh=CONF_THRESH
    vis_all_cls(im, image_name, output_bbox_dict)
    vis_gt(im, image_name,json_gt,out_dir)




# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
#     parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
#                         choices=NETS.keys(), default='res101')
#     parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
#                         choices=DATASETS.keys(), default='ban')
#     parser.add_argument("--ckpt", help="Check point file", type=str, required=True)
#     parser.add_argument('-v', '--version', dest='version', help='1 2', default=2, type=int)
#     args = parser.parse_args()
#
#     return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # args = parse_args()
    import os


    demonet = "mobile"
    tfmodel = "mobile_faster_rcnn_iter_9000.ckpt"
    json_gt = "/data/ubuntu/qiaoran/mobile_net_faster_rcnn/data/CC_YanZhenQ_val/annotations/annotation.json"

    out_dir = "/data/ubuntu/qiaoran/mobile_net_faster_rcnn/inerence_orginal_images_out/"
    folder = '/data/ubuntu/qiaoran/mobile_net_faster_rcnn/inerence_orginal_images/'

    CLASSES = ('__background__', 'CC_YanZhenQ')

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(num_layers=50)
    elif demonet == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    for im_name in os.listdir(folder):
        im_name = os.path.join(folder, im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name,json_gt)
        # exit(0)
