# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import json
from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms
import shutil

def vis_gt(im, data_folder, img_name):

  json_path = os.path.join(data_folder, 'annotations/annotation.json')

  with open(json_path, 'r') as file:
    data = json.load(file)


  for item in data['images']:
    if item['file_name'] == img_name:
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

  return im


def vis_detections(im, class_name, dets,imgName, thresh=0.5):

  """Visual debugging of detections."""
  fw = open('result.txt','a')

  for i in range(np.minimum(10, dets.shape[0])):

    bbox = tuple(int(np.round(x)) for x in dets[i, :4])

    score = dets[i, -1]

    if score > thresh:

      cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
      fw.write(('%s %s:%.3f\n' % (imgName,class_name, score)))

      cv2.putText(im,'%s:%.3f' % (class_name, score), (bbox[0]+15, bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,2.0, (0, 255, 255), thickness=1)
  fw.close()
  return im

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            # interpolation=cv2.INTER_LINEAR)
            interpolation=cv2.INTER_AREA)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  # print('*************')
  # print(blobs['data'].shape)
  # print(blobs['data'])
  # import random
  # ran_num = random.random()
  # cv2.imwrite('/home/qiaoran/Project/mobilenet_fastercnn/inerence_cal_test_check/5_15_DD/check_input/im_check{}.jpg'.format(ran_num), blobs['data'][0])
  # print('*************')

  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))


  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, image_folder, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  shutil.rmtree(image_folder)
  os.mkdir(image_folder)

  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in range(num_images):


    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    im2show = np.copy(im)
    gt_im2show = np.copy(im)

    imgName = imdb.image_path_at(i).split('/')[-1]
    gtimgName = imgName.split('.')[0] + '_gt.' + imgName.split('.')[-1]

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets
      im2show = vis_detections(im2show, imdb.classes[j], cls_dets,imgName)

      im2show_gt = vis_gt(gt_im2show, imdb._data_path, imgName)

      cv2.imwrite(os.path.join(image_folder,imgName), im2show)
      cv2.imwrite(os.path.join(image_folder,gtimgName), im2show_gt)


    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()


    print('{} im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(imgName, i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time), end="\r")
  print()
  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

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

def crop_image(image, crop_size, step_size):
    # print("crop size", crop_size, "step_size", step_size)
    h, w, _ = image.shape
    crop_poses = cropImageSize((w, h), crop_size, step_size)
    cropped_images = []
    for crop_rect in crop_poses:
        crop_img = crop_rect.crop_img(image)
        cropped_images.append((crop_img, crop_rect))
    return cropped_images

def clean_inside_boxes(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def test_net_cropped(sess, image_folder, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  shutil.rmtree(image_folder)
  os.mkdir(image_folder)

  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  crop_size = (1024, 1024)
  step_size = (500, 500)

  for i in range(num_images):


    im = cv2.imread(imdb.image_path_at(i))
    cropped_images = crop_image(im, crop_size=crop_size, step_size=step_size)
    _t['im_detect'].tic()

    total_scores = []
    total_boxes = []
    for sub_img, rect in cropped_images[:]:
      scores, boxes = im_detect(sess, net, sub_img)

      for ind in range(1, imdb.num_classes):
        boxes[:, 4 * ind] += rect.x_min
        boxes[:, 4 * ind + 1] += rect.y_min
        boxes[:, 4 * ind + 2] += rect.x_min
        boxes[:, 4 * ind + 3] += rect.y_min
      total_scores.append(scores)
      total_boxes.append(boxes)
    scores = np.concatenate(tuple(total_scores), axis=0)
    boxes = np.concatenate(tuple(total_boxes), axis=0)
    _t['im_detect'].toc()

    _t['misc'].tic()

    im2show = np.copy(im)
    gt_im2show = np.copy(im)

    imgName = imdb.image_path_at(i).split('/')[-1]
    gtimgName = imgName.split('.')[0] + '_gt.' + imgName.split('.')[-1]

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]

      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]

      # clean boxe_inside_box
      keep = clean_inside_boxes(cls_dets, 0.55)
      cls_dets = cls_dets[keep, :]
      # print(cls_dets)

      all_boxes[j][i] = cls_dets
      im2show = vis_detections(im2show, imdb.classes[j], cls_dets,imgName)

      im2show_gt = vis_gt(gt_im2show, imdb._data_path, imgName)

      cv2.imwrite(os.path.join(image_folder,imgName), im2show)
      cv2.imwrite(os.path.join(image_folder,gtimgName), im2show_gt)


    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()


    print('{} im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(imgName, i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time), end="\r")
  print()
  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)
