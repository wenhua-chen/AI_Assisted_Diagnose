# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import uuid

import numpy as np
import os.path as osp
import scipy.sparse
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import datasets.ds_utils as ds_utils
from datasets.imdb import imdb
from model.config import cfg




def _do_detection_eval(res_file, _COCO):
    ann_type = 'bbox'
    print("custom ds res file: ", res_file)
    coco_dt = _COCO.loadRes(res_file)
    coco_eval = COCOeval(_COCO, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.params.iouThrs = np.linspace(0.3, 0.95, np.round((0.95 - 0.3) / 0.05) + 1, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    _custom_eval_detection(coco_eval)

def _custom_eval_detection(cocoeval, log=print):
    cocoGt = cocoeval.cocoGt
    cocoDt = cocoeval.cocoDt

    accumulate = dict()
    result = dict()

    img_counts, area_counts, cat_counts = len(cocoeval.params.imgIds), len(cocoeval.params.areaRng), len(cocoeval.params.catIds)
    log("==" * 40)
    for cat_i in range(cat_counts):
        eval_counts = area_counts * img_counts
        for eval_i in range(img_counts):
            # only check full size area
            idx = cat_i * eval_counts + eval_i
            eval_result = cocoeval.evalImgs[idx]
            if eval_result is None:
                continue
            dt_result = [(id, score) for id, score in zip(eval_result["dtIds"], eval_result["dtScores"]) if score > 0.5]
            dt_count = len(dt_result)
            gt_count = len(eval_result["gtIds"])
            tp_count = 0
            for i, gtId in enumerate(eval_result["gtIds"]):
                matches = eval_result["gtMatches"][:, i][::-1]
                dtId = 0
                for _id in matches:
                    if _id != 0:
                        dtId = _id
                        break
                if dtId != 0:
                    if cocoDt.anns[dtId]["score"] > 0.5:
                        tp_count += 1
            accumulate[(cat_i, eval_i)] = {"gt_count": gt_count, "dt_count": dt_count, "true_positive_count": tp_count}

    # per category:
    result["category"] = dict()
    for cat_i in range(cat_counts):
        recall = []
        fake_pos = []
        empty_count = 0
        for img_i in range(img_counts):
            if (cat_i, img_i) in accumulate:
                info = accumulate[(cat_i, img_i)]
                if info["gt_count"] > 0:
                    recall.append(info["true_positive_count"] / info["gt_count"])
                    if info["dt_count"] == 0:
                        empty_count += 1
                if info["dt_count"] != 0:
                    fake_pos.append(float((info["dt_count"] - info["true_positive_count"]) / info["dt_count"]))

        _recall = np.mean(recall)
        _fake_pos = np.mean(fake_pos)
        cat_id = cocoeval.params.catIds[cat_i]
        result["category"][cocoeval.cocoGt.cats[cat_id]["name"]] = [_recall, _fake_pos, empty_count / len(cocoeval.evalImgs)]
        log("category {}: recall: {:.3f}, fake positive: {:.3f}, found no target: {:.3f}".
            format(cocoeval.cocoGt.cats[cat_id]["name"], _recall, _fake_pos, empty_count / len(cocoeval.evalImgs)))
    log("==" * 40)
    total_recall = []
    total_fake_pos = []

    # per images:
    result["image"] = dict()
    for img_i in range(img_counts):
        tp_count = 0
        dt_count = 0
        gt_count = 0

        for cat_i in range(cat_counts):
            if (cat_i, img_i) in accumulate:
                info = accumulate[(cat_i, img_i)]
                tp_count += info["true_positive_count"]
                gt_count += info["gt_count"]
                dt_count += info["dt_count"]

        _recall = tp_count / gt_count
        if dt_count != 0:
            _fake_pos = (dt_count - tp_count) * 1.0 / dt_count
        else:
            _fake_pos = 0

        total_recall.append(_recall)
        total_fake_pos.append(_fake_pos)
        img_id = cocoeval.params.imgIds[img_i]
        cocoGt.imgs[img_id]["file_name"]
        result["image"][cocoGt.imgs[img_id]["file_name"]] = [_recall, _fake_pos]
        log("image {}: recall: {:.3f}, fake positive: {:.3f}".
            format(cocoeval.params.imgIds[img_i], _recall, _fake_pos))
    log("==" * 40)
    result["total"] = [np.mean(total_recall), np.mean(total_fake_pos)]
    log("recall: {:.3f}, fake positive: {:.3f}".
        format(np.mean(total_recall), np.mean(total_fake_pos)))
    return result
    

if __name__=='__main__':
    val_json = '/home/lyc/tf-faster-rcnn/data/ban_new_val/annotations/annotaion.json'
    res_file = '/home/lyc/tf-faster-rcnn/output/mobile/ban_new_val/default/'
    _COCO = COCO(val_json)
    _do_detection_eval(res_file, _COCO)