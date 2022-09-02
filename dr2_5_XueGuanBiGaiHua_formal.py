import os

import numpy as np
import pandas as pd
import cv2
import math

import mmcv
import torch

from mmdet.apis import init_detector, inference_detector
from shapely.geometry import Point, Polygon

def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def cal_intersect(boxA, boxB):
    if len(boxA)==0 or len(boxB)==0: return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return max(interArea/boxBArea, interArea/boxAArea)

def drop_inside_bboxes(boxes, clss,iou_the=0.3):
    '''
    boxes: [[2187, 666, 2238, 723, 0.56], [2070, 602, 2185, 759, 0.56], [572, 1462, 709, 1588, 0.7], [2312, 1564, 2482, 1671, 0.9]]
    cls = ['BanPianYing', 'BanPianYing', 'BanPianYing', 'BanPianYing']
    '''

    assert len(boxes) == len(clss)
    d = {}
    for i in range(len(boxes)):
        d[str(boxes[i][0:4])] = clss[i]

    original_bboxes = boxes[:]
    boxes.sort(key=lambda x: x[4], reverse=True)
    group = np.array(boxes)

    out_array = []
    out_cls = []
    while len(group) != 0:
        del_idx = []
        for i, item in enumerate(group):
            min_intersection_ratio = cal_intersect(group[0][0:4], group[i][0:4])
            if min_intersection_ratio >= iou_the:
                del_idx.append(i)
        out_array.append([x for x in group[0][0:4].tolist()] + [group[0][-1]])
        group = np.delete(group, del_idx, axis=0)

    for element in out_array:
        out_cls.append(d[str(element[0:4])])
    return out_array, out_cls


def calculate_intersection(coords_big,coords2_small,theath=0.3):
    # cords = trans_box(coords2_small,full_boxes=False)
    p1 = Polygon(coords_big)
    p2 = Polygon(coords2_small)
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    inter_sec_area = p1.intersection(p2).area

    overlap1 = inter_sec_area/p1.area
    overlap2 = inter_sec_area/p2.area
    if overlap1 > overlap2:
        overlap = overlap1
    else:
        overlap = overlap2
    if overlap > theath:
        return 'True'
    else:
        return 'False'

def trans_box(boxes,full_boxes=True):
    '''
    full_boxes means a list of box
    input:
        [[7.69001038e+02 1.44156030e+03 8.20211121e+02 1.48525391e+03 6.36455059e-01]
            [8.27617493e+02 1.53489282e+03 9.54969788e+02 1.62012524e+03 8.95488933e-02]]
    output:
        [[(769.001038, 1441.5603),(820.211121, 1441.5603),
          (769.001038, 1485.25391),(820.211121, 1485.25391)],
         [(827.617493, 1534.89282),(954.969788, 1534.89282),
          (827.617493, 1620.12524),(954.969788, 1620.12524)]]
    '''
    if full_boxes:
        boxes_4cords = []
        for box in boxes:
            box_i = []
            pts = box[:4]
            xmin,ymin,xmax,ymax = pts[0],pts[1],pts[2],pts[3]

            p1=(xmin,ymin)
            p2=(xmax,ymin)
            p3=(xmax,ymax)
            p4=(xmin,ymax)

            box_i.append(p1)
            box_i.append(p2)
            box_i.append(p3)
            box_i.append(p4)
            boxes_4cords.append(box_i)
        return boxes_4cords
    else:
        box_i = []
        pts = boxes[:4]
        xmin,ymin,xmax,ymax = pts[0],pts[1],pts[2],pts[3]
        p1=(xmin,ymin)
        p2=(xmax,ymin)
        p3=(xmax,ymax)
        p4=(xmin,ymax)
        box_i.append(p1)
        box_i.append(p2)
        box_i.append(p3)
        box_i.append(p4)
        return box_i

def fei_within_filter(QiGuan_return,boxes,cls_str):

    coords_big0 = QiGuan_return['fei0_contour']
    coords_big1 = QiGuan_return['fei1_contour']
    boxes_4cords = trans_box(boxes,full_boxes=True)

    del_list = []
    for i in range(len(boxes_4cords)):
        TorF0 = calculate_intersection(coords_big0,boxes_4cords[i],theath=0.5)
        TorF1 = calculate_intersection(coords_big1,boxes_4cords[i],theath=0.5)
        if TorF0 == 'True' or TorF1 == 'True':
            del_list.append(i)
        else:
            pass

    boxes = [i for j, i in enumerate(boxes) if j not in del_list]
    cls_str = [i for j, i in enumerate(cls_str) if j not in del_list]
    return boxes,cls_str


def pred(model,img_path,pts,thresh=0.1,iou_s=0.1,qiguan_return=None):

    cls_names = ['XGBGH', 'WYY']
    img_original = cv2.imread(img_path)
    img_crop = img_original[pts[1]:pts[3],pts[0]:pts[2]]
    result = inference_detector(model, img_crop)
    bbox_result, segm_result = result, None
    bboxes_np = np.vstack(bbox_result)
    labels_np = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
        ]
    labels_np = np.concatenate(labels_np)

    inds = np.where(bboxes_np[:, -1] >= thresh)[0]
    boxes_s = bboxes_np[inds, :].tolist()

    clses_idx = labels_np[inds].tolist()
    clses = [cls_names[i] for i in clses_idx]

    boxes_s_original, clses_original = drop_inside_bboxes(boxes_s, clses,iou_the=iou_s)
    boxes_s = []
    clses = []
    for i, clse in enumerate(clses_original):
        if clse not in ['WYY','meaningless']:
            boxes_s.append(boxes_s_original[i])
            clses.append(clses_original[i])

    boxesr = []
    for box in boxes_s:
        boxesr.append([int(box[0]+pts[0]),int(box[1]+pts[1]),int(box[2]+pts[0]),int(box[3]+pts[1]),truncate(box[4], 3)])

    if qiguan_return:
        boxesr,clses = fei_within_filter(qiguan_return,boxesr,clses)
    return boxesr, clses

def vis(img_path, boxes_nms, cls_name, score=0.1):
    im = cv2.imread(img_path)
    for box_nms in boxes_nms:
        if box_nms[4] < score:
            continue
        cv2.putText(im, cls_name+'_'+str(box_nms[4]), (box_nms[0], box_nms[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(im, (box_nms[0], box_nms[1]), (box_nms[2], box_nms[3]), (0,0,255),2)
    img_name = os.path.basename(img_path)
    cv2.imwrite('{}_{}_pred.jpg'.format(img_name.split('.')[0], cls_name), im)

if __name__ == '__main__':
    from timeit import default_timer as timer
    import json
    import shutil
    from tqdm import tqdm
    from pycocotools.coco import COCO
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    config_file = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_5_XueGuanBiGaiHua_formal_cfg.py'
    pth_model = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_5_XueGuanBiGaiHua_formal.pth'

    start = timer()
    model = init_detector(config_file, pth_model, device='cuda:0')
    elapsed_time = round(timer() - start,2)
    print('{} model loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))

    single_test = True
    if single_test:
        # im_name = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0_single_test_img/PN038159.jpg'
        # im_name = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/0_test_img_original/PN038161.jpg'
        # im_name = '/data/steven/project/Object_Detection_coastal/dataser_raw/0_All_orginal_image_real/orginal_img/PN019926.jpg'
        im_name = '/data/steven/project/Object_Detection_coastal/dataser_raw/0_All_orginal_image_real/orginal_img/PN029015.jpg'
        pts1 = [1301, 894, 1847, 1628]
        start = timer()

        boxesr, clses = pred(model,im_name,pts1)
        print('_'*100)
        print(boxesr, clses)
        vis(im_name, boxesr, 'xueguanbi')

        elapsed_time = round(timer() - start,2)
        print('{} model pred {}s {}'.format('-'*20,elapsed_time,'-'*20))
