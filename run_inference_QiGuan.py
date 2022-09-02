# -*- coding:utf-8 -*- 
# Author: 陈文华(Steven)
# Website: https://wenhua-chen.github.io/
# Github: https://github.com/wenhua-chen
# Date: 2020-03-16 05:32:10
# LastEditTime: 2022-09-02 15:53:01
# Description: 

import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import shutil

def get_cls_name(json_file,return_en=True):
    '''
    ['B_QBHQSZ','Z_Z','CC_QZNB','SSX_SSJT']
    '''
    with open(json_file, "r", encoding="utf-8") as f:
        keyword_dict = json.load(f)
    catagerts_names = list(keyword_dict.keys())
    list_tmp = []
    for k,v in keyword_dict.items():
        list_i = [k,v[1],v[0]]
        list_tmp.append(list_i)
    list_tmp = sorted(list_tmp,key=lambda x: x[2], reverse=False)
    en_name=[x[1] for x in list_tmp]
    cn_name=[x[0] for x in list_tmp]
    # en_name.insert(0, "background")
    # cn_name.insert(0, "background")
    if return_en:
        return en_name
    else:
        return cn_name


def trans2caffe2(result,cls_names):
    n_cls = len(cls_names)
    if 'background' not in cls_names:
        n_cls+=1

    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = mmcv.concat_list(segm_result)

    print('bboxes',bboxes)
    print('')
    print('segms',segms)
    print('')
    print('labels',labels)
    print('')

    c_boxes,c_segms,c_keypoints = [],[],None
    for i in range(n_cls):
        c_boxes.append([])
        c_segms.append([])

    for j in range(len(labels)):
        idx = labels[j]
        idx+=1

        # box = np.array(bboxes[i]).astype('float32')
        box = bboxes[j]
        seg = segms[j]

        c_boxes[idx].append(box)
        c_segms[idx].append(seg)

    c_boxes_np = []
    for k in c_boxes:
        if len(k)>0:
            c_boxes_np.append(np.array(k).astype('float32'))
        else:
            c_boxes_np.append(k)

    print('c_boxes_np',c_boxes_np)
    print('')
    print('c_segms',c_segms)
    print('')
    print('c_keypoints',c_keypoints)
    print('')
    return c_boxes_np,c_segms,c_keypoints

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = cv2.imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        cv2.imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(img, out_file)


def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.7 + color_mask * 0.3
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # mmcv.imshow_det_bboxes(
    #     img.copy(),
    #     bboxes,
    #     labels,
    #     class_names=class_names,
    #     score_thr=score_thr,
    #     show=out_file is None,
    #     wait_time=wait_time,
    #     out_file=out_file)
    font_scale = 0.5
    text_color = (200,0,0)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        # cv2.rectangle(
        #     img, left_top, right_bottom, bbox_color, thickness=1)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    cv2.imwrite(out_file,img)

if __name__ == "__main__":
    from timeit import default_timer as timer
    import json
    from tqdm import tqdm
    from mmdet.apis import init_detector, inference_detector
    from pycocotools.coco import COCO


    try_time = '1011_ZuiTi_QiGuan_960_T0'
    imgFolder = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/3_ruwan_hostipal_data_1121/1_false_negtive/'
    keyword_json = '/data/steven/project/Object_Detection_coastal/dataser_raw/10_QiGuan_series/COCOjson/ZuiTi_QiGuan_4cls_keyword.json'

    config_file = './DR_models_configs/dr1_0_QiGuan_formal_cfg.py'
    checkpoint_file = './DR_models_configs/dr1_0_QiGuan_formal.pth'
    # out_dir = './QiGuan_draw/'
    out_dir = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/3_ruwan_hostipal_data_1121/1_false_negtive_pred_qiguan/'

    # config_file = '/data/steven/project/Object_Detection_coastal/mmdetection_project/saved_cfgs/4_QiGuan_Series/1011_ZuiTi_QiGuan_960_T0/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
    # checkpoint_file = '/data/steven/project/Object_Detection_coastal/mmdetection_project/output/10_QiGuan_series/ZuiTi_QiGuan_dataset_1011_ZuiTi_QiGuan_960_T0/cascade_mask_rcnn_hrnetv2p_w32_20e/epoch_25.pth'

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    start = timer()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    elapsed_time = round(timer() - start,2)
    # print('model_name:{}',format(model_name))
    print('{} model loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))
    start = timer()
    cls_names = get_cls_name(keyword_json,return_en=True) #must en class name
    print('cls_names',cls_names)

    img_list =os.listdir(imgFolder)
    for i in tqdm(range(len(img_list))):
        im_name = img_list[i]
        in_path = os.path.join(imgFolder, im_name)
        out_path = os.path.join(out_dir, im_name)

        assert os.path.exists(in_path)
        thresh = 0.3

        result = inference_detector(model, in_path)
        # boxes, segms, keypoints = trans2caffe2(result,cls_names)
        # break

        elapsed_time = round(timer() - start,2)

        # print('inference {}s'.format(elapsed_time))

        show_result(in_path, result, model.CLASSES,score_thr=thresh,out_file=out_path)
        # print('{} done'.format(out_path))
