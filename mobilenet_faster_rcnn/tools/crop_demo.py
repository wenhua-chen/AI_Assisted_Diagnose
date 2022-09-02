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
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import math

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from nets.mobilenet_v2 import mobilenetv2
import json

import tensorflow as tf
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="3"



# CLASSES = ('__background__', 'ban')
CLASSES = ('__background__', 'huangheban', 'douyin', 'qitaban')

NETS = ["vgg16", "res50", "res101", "mobile"]

def vis_detections(image, class_name, dets, img_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print("found no target")
        return

    im = image[:, :, (2, 1, 0)].copy()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print(bbox)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        im = cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,
            2, (255, 255, 255), 2, cv2.LINE_AA)
    folder = os.path.join(os.path.dirname(img_name), "result")
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_file = os.path.join(folder, "detections_" + os.path.basename(img_name))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # print('im.shape', im.shape)
    # print('save_file', save_file)
    cv2.imwrite(save_file, im)
    print ("save image to "+save_file)

    # im = image[:, :, (2, 1, 0)].copy()
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # for i in inds:
    #     bbox = dets[i, :4]
    #     score = dets[i, -1]
    #     bbox = list(map(int, bbox))
    #     bbox_file = str(img_name)+"_"+str(i)+"_bbox.jpg"
        #cv2.imwrite(bbox_file, im[bbox[1]:bbox[3], bbox[0]:bbox[2], :])

class Rect(object):
    def __init__(self, x, y, w, h):
        self.x_min = x
        self.y_min = y
        self.w = w
        self.h = h
        self.x_max = x + w
        self.y_max = y + h
    def print(self):
        print("x {}, y {}, w {}, h {}, x_max {}, y_max {}".format(self.x_min, self.y_min, self.w, self.h,
            self.x_max, self.y_max
        ))
    def crop_img(self, img):
        return img[self.y_min:self.y_max, self.x_min:self.x_max, :]
    def crop_bbox(self, bbox, bbox_min_size=200, min_w=50, min_h=50):
        b_x, b_y, b_w, b_h = bbox
        _lx = max(self.x_min, b_x)
        _ly = max(self.y_min, b_y)
        _rx = min(self.x_max, b_x + b_w)
        _ry = min(self.y_max, b_y + b_h)
        if _lx < _rx and _ly < _ry:
            n_w, n_h = _rx-_lx, _ry-_ly
            if n_w * n_h > bbox_min_size and n_w > min_w and n_h > min_h:
                return [_lx, _ly, _rx-_lx, _ry-_ly]
            elif n_w * n_h / (b_w * b_h) > 0.8:
                return [_lx, _ly, _rx-_lx, _ry-_ly]
            else:
                return None
        else:
            return None
    def generate_new_name(self, old_name):
        pre_names, ext = os.path.splitext(old_name)
        pre_names += "_{}_{}_{}_{}".format(self.x_min, self.y_min, self.w, self.h)
        return "".join([pre_names, ext])

def cropImageSize(ori_size, crop_size, step_size):
    crop_results = []
    if crop_size[0] > ori_size[0] or crop_size[1] > ori_size[1] or step_size[0] <= 0 or step_size[1] <= 0:
        print("Error crop parameters. ", ori_size, step_size, crop_size)
        return crop_results
    x_steps = math.ceil((ori_size[0] - crop_size[0]) / step_size[0] + 1)
    y_steps = math.ceil((ori_size[1] - crop_size[1]) / step_size[1] + 1)
    for i in range(int(x_steps)):
        x_pos =  min(i * step_size[0], ori_size[0] - crop_size[0])
        for j in range(int(y_steps)):
            y_pos = min(j * step_size[1], ori_size[1] - crop_size[1])
            crop_results.append(Rect(x_pos, y_pos, crop_size[0], crop_size[1]))
    return crop_results

def vis_gt(image, img_name, json_path='/home/lyc/tf-faster-rcnn/data/ban_new_val/annotations/annotation.json'):
    img_name_find = img_name.split('/')[-1]
    save_name = img_name.split('.')[0] + '_gt.jpg'
    save_name = os.path.dirname(save_name) + '/result/' + os.path.basename(save_name)
    print('save_name_gt', save_name)

    # json_path = os.path.join(data_folder, 'annotations/annotation.json')
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

def crop_image(image, crop_size, step_size):
    print("crop size", crop_size, "step_size", step_size)
    h, w, _ = image.shape
    crop_poses = cropImageSize((w, h), crop_size, step_size)
    cropped_images = []
    for crop_rect in crop_poses:
        crop_img = crop_rect.crop_img(image)
        cropped_images.append((crop_img, crop_rect))
    return cropped_images

def vis_all_cls(image, image_name, output_dict):
    # print('image_name', image_name)
    save_name = image_name.split('.')[0] + '_pred.jpg'
    save_name = os.path.dirname(save_name) + '/result/' + os.path.basename(save_name)

    # print('save_name', save_name)
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

def demo(sess, net, image_name, crop_size, step_size):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)
    cropped_images = crop_image(im, crop_size, step_size)

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    total_scores = []
    total_boxes = []

    for sub_img, rect in cropped_images[:]:
        scores, boxes = im_detect(sess, net, sub_img)

        for ind in range(len(CLASSES[1:])):
            ind += 1
            cls_boxes = boxes[:,4*ind:4*(ind+1)]
            cls_scores = scores[:, ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # vis_detections(sub_img, "ban", dets, rect.generate_new_name(image_name), thresh=CONF_THRESH)
            boxes[:, 4*ind] += rect.x_min
            boxes[:, 4*ind+1] += rect.y_min
            boxes[:, 4*ind+2] += rect.x_min
            boxes[:, 4*ind+3] += rect.y_min
        
        exit(0)
        total_scores.append(scores)
        total_boxes.append(boxes)

    timer.toc()

    scores = np.concatenate(tuple(total_scores), axis=0)
    boxes = np.concatenate(tuple(total_boxes), axis=0)
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))


    output_bbox_dict = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds, :]
        output_bbox_dict[cls] = dets

        # vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)

    vis_all_cls(im, image_name, output_bbox_dict)
    vis_gt(im, image_name)

def demo_nms_before(sess, net, image_name, crop_size, step_size):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)
    cropped_images = crop_image(im, crop_size, step_size)

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.2

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    total_scores = []
    total_boxes = []
    total_dets = []
    for sub_img, rect in cropped_images[:]:
        scores, boxes = im_detect(sess, net, sub_img)
        for ind in range(len(CLASSES[1:])):
            ind += 1
            boxes[:, 4*ind] += rect.x_min
            boxes[:, 4*ind+1] += rect.y_min
            boxes[:, 4*ind+2] += rect.x_min
            boxes[:, 4*ind+3] += rect.y_min
            cls_boxes = boxes[:,4*ind:4*(ind+1)]
            cls_scores = scores[:, ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detections(sub_img, "ban", dets, rect.generate_new_name(image_name), thresh=CONF_THRESH)
        total_scores.append(scores)
        total_boxes.append(boxes)
    timer.toc()
    scores = np.concatenate(tuple(total_scores), axis=0)
    boxes = np.concatenate(tuple(total_boxes), axis=0)
    dets = np.concatenate(tuple(total_dets), axis=0)
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    vis_detections(im, "ban", dets, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS, default='res101')
    parser.add_argument("--ckpt", help="Check point file", type=str, required=True)
    parser.add_argument('-v', '--version', dest='version', help='1 2', default=2, type=int)
    parser.add_argument("--crop_size", "-c", default=1024, type=int, help="image crop size")
    parser.add_argument("--step_size", "-s", default=500, type=int, help="stride step size")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    tfmodel = args.ckpt

    crop_size = (args.crop_size, args.crop_size)
    step_size = (args.step_size, args.step_size)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    # tfconfig = tf.ConfigProto(allow_soft_placement=False)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True

    # # init session
    # configGpu = tf.ConfigProto()
    # configGpu.gpu_options.allow_growth=True
    # sess = tf.Session(config=configGpu)

    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(num_layers=50)
    elif demonet == 'mobile':
        if args.version == 1:
            net = mobilenetv1()
        else:
            net = mobilenetv2()
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 4,
                          tag='default', anchor_scales=[4,8,16,32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # folder = os.path.join(cfg.DATA_DIR, 'demo')
    # im_names = os.listdir(folder)


    folder = '/home/lyc/tf-faster-rcnn/images_ban_crop/'
    im_names = os.listdir(folder)

    for im_name in im_names:
        if ".jpg" in im_name:
            print('Demo for data/demo/{}'.format(im_name))
            demo(sess, net, os.path.join(folder, im_name), crop_size, step_size)
        # exit(0)
