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


class custom_ds(imdb):
    def __init__(self, dataset_name):
        super(custom_ds, self).__init__(dataset_name)
        # COCO specific config options
        self.config = {'use_salt': True,
                       'cleanup': True}
        # name, paths
        self._data_path = osp.join(cfg.DATA_DIR, dataset_name)
        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                                   self._COCO.getCatIds())))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)

        self._data_name = ""
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def _get_ann_file(self):

        return osp.join(self._data_path, 'annotations', "annotation.json")

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self._COCO.imgs[index]["file_name"]
        image_path = osp.join(self._data_path, 'images',
                              file_name)
        assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self._image_index]

        parent_path = os.path.abspath(os.path.join(cache_file, os.pardir))
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        print('len(widths)',len(widths))
        print('num_images',num_images)
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas']}

            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        # print('~~~~ Summary metrics ~~~~')
        # coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        print("custom ds res file: ", res_file)
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.params.iouThrs = np.linspace(0.3, 0.95, np.round((0.95 - 0.3) / 0.05) + 1, endpoint=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        result_data = self._custom_eval_detection(coco_eval)
        self._print_detection_eval_metrics(coco_eval)
        eval_file = osp.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

        eval_json_file = osp.join(output_dir, 'eval_json_file.json')
        with open(eval_json_file, "w") as f:
            json.dump(result_data, f, indent=4, sort_keys=True)
    # def _custom_eval_detection(self, cocoeval):
    #     cocoGt = cocoeval.cocoGt
    #     cocoDt = cocoeval.cocoDt
    #     recall = []
    #     fake_pos = []
    #     empty_count = 0
    #     for eval_result in cocoeval.evalImgs:
    #         dt_result = [score for score in eval_result["dtScores"] if score > 0.5]
    #         dt_count = len(dt_result)
    #         gt_count = len(eval_result["gtIds"])
    #         tp_count = 0

    #         for i, gtId in enumerate(eval_result["gtIds"]):
    #             matches = eval_result["gtMatches"][:, i][::-1]
    #             dtId = 0
    #             for _id in matches:
    #                 if _id != 0:
    #                     dtId = _id
    #                     break
    #             if dtId != 0:
    #                 if cocoDt.anns[dtId]["score"] > 0.5:
    #                     tp_count += 1
    #         recall.append(float(tp_count / gt_count))
    #         if dt_count != 0:
    #             fake_pos.append(float((dt_count - tp_count) / dt_count))
    #         else:
    #             empty_count += 1
    #     _recall = np.mean(recall)
    #     _fake_pos = np.mean(fake_pos)
    #     print("recall: {}, fake positive: {}, found no target: {}".
    #           format(_recall, _fake_pos, empty_count / len(cocoeval.evalImgs)))

    def _custom_eval_detection(self, cocoeval, log=print):
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
        # print('^^^^^^'*40)
        # print(result)
        # print('^^^^^^'*40)

        return result

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes - 1))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._name +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'

        # res_file = '/home/lyc/tf-faster-rcnn/output/mobile/ban_new_val/default/mobile_faster_rcnn_iter_102000/detections_ban_new_val_results_7f4dcd45-1139-403a-bb50-90089b6d5e31.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        # if self._image_set.find('test') == -1:
        self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
