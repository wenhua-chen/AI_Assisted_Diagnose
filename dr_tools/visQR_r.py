# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append('/home/steven/mask_rcnn_caffe2/Detectron-xRay/lib/utils/')
import cv2
import numpy as np
import os
import re
import pycocotools.mask as mask_util

from utils.colormap import colormap
import utils.env as envu
# import utils.keypoints as keypoint_utils

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        show_box=False,fill_mask=False,
        ext='',generate_mask=False,mask_dir = None):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return


    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0

    colors_list = [
              (0, 0, 0),
              (1, 0, 0.2),
              (0.3, 0.3, 1),
              (0, 0.3, 0.9),
              (0.9, 0.5, 0.3),
              (0.9, 0.3, 1),
              (0.8, 0.4, 0.3),
              (0.6, 0.9, 0.2),
              (0.6, 0.2, 0.9),
              (0.8, 0.3, 0),
              (0.3, 0.8, 0.7),
              (0.9, 0.9, 1),
              (1, 0.8, 0.4),
              (0.5, 0.1, 0.9),
              (1, 0.5, 1),
              (0.5, 0.8, 1),
              ]

    fei_idx = 0
    mask_dict = {}
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        cls = classes[i]
        # print(cls)
        cls_color = colors_list[cls]
        class_text = dataset.classes[cls]

        if score < thresh:
            continue

        if show_box:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False, edgecolor='r',
                              linewidth=2, alpha=box_alpha))

        if show_class:
            # if class_text in ['fei','xinying']:
            ax.text(
                bbox[2]+8, bbox[1] + 80,
                get_class_string(classes[i], score, dataset),
                fontsize=18,
                family='serif',
                bbox=dict(
                    # facecolor='b', alpha=0.4, pad=5, edgecolor='none'),
                    facecolor=cls_color, alpha=0.4, pad=5, edgecolor='none'),
                color='white')
        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            #print(e[e!=0])
            #print(classes[i])
            # print('mask_dir',mask_dir)
            if generate_mask:
                e[e!=0]=255
                file_name =os.path.basename(im_name)
                filename, file_extension = os.path.splitext(file_name)
                class_text = dataset.classes[cls]
                # print('class_text',class_text)
                if class_text == 'fei':
                    # print('fei_idx',fei_idx)
                    output_name = filename+'_'+class_text+str(fei_idx)+'.jpg'
                    mask_dict[class_text+str(fei_idx)]=mask_dir+output_name
                    fei_idx+=1
                else:
                    output_name = filename+'_'+class_text+'.jpg'
                    mask_dict[class_text]=mask_dir+output_name
                if class_text not in ['__background__','qiguan']:
                # if class_text not in ['__background__','qiguan','jizhu','xinying']:
                    mask_path = mask_dir+output_name
                    cv2.imwrite(mask_path,e)
                    # print(mask_path)
            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                if fill_mask:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor='w', linewidth=1.2,
                        alpha=0.5)
                else:
                    # if class_text in ['fei','xinying']:
                    polygon = Polygon(
                            c.reshape((-1, 2)),
                            fill=False, facecolor='none',
                            edgecolor=cls_color, linewidth=3,
                            alpha=0.5)
                    ax.add_patch(polygon)

    # output_name = os.path.basename(im_name) + '.' + ext
    output_name = os.path.basename(im_name)
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi,quality=20)
    plt.close('all')

    return mask_dict
