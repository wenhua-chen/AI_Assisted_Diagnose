import os
import numpy as np
import pandas as pd
import cv2
import math
import mmcv
import torch
from mmdet.apis import init_detector, inference_detector


def byte2readable(size,precision=2):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])


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

def exapnd_upDown_porcess(input_img,cord_i,expand_lung=True,upDown=False):
    img_h, img_w, img_c = input_img.shape

    if cord_i[0][0] < cord_i[1][0]:
        Lx1,Ly1,Lx2,Ly2,aw,ah = cord_i[0][0],cord_i[0][1],cord_i[0][2],cord_i[0][3],cord_i[0][2]-cord_i[0][0],cord_i[0][3]-cord_i[0][1]
        Rx1,Ry1,Rx2,Ry2,bw,bh = cord_i[1][0],cord_i[1][1],cord_i[1][2],cord_i[1][3],cord_i[1][2]-cord_i[1][0],cord_i[1][3]-cord_i[1][1]
    else:
        Lx1,Ly1,Lx2,Ly2,aw,ah = cord_i[1][0],cord_i[1][1],cord_i[1][2],cord_i[1][3],cord_i[1][2]-cord_i[1][0],cord_i[1][3]-cord_i[1][1]
        Rx1,Ry1,Rx2,Ry2,bw,bh = cord_i[0][0],cord_i[0][1],cord_i[0][2],cord_i[0][3],cord_i[0][2]-cord_i[0][0],cord_i[0][3]-cord_i[0][1]

    wid_r = 0.1
    het_r = 0.07
    if expand_lung:
        Lx1 = int(Lx1-wid_r*aw)
        Ly2 = int(Ly2+het_r*ah)
        Ly1 = int(Ly1-het_r*ah)

        if Lx1 < 0:
            Lx1 = 1
        if Ly1 < 0:
            Ly1 = 1
        if Ly2 > img_h:
            Ly2 = img_h-1

        Rx2 = int(Rx2+wid_r*aw)
        Ry2 = int(Ry2+het_r*ah)
        Ry2 = int(Ry2-het_r*ah)
        if Ry2 < 0:
            Ry2 = 1
        if Rx2 > img_w:
            Rx2 = img_w-1
        if Ry2 > img_h:
            Ry2 = img_h-1
    assert Lx1<Lx2,Ly1<Ly2
    assert Rx1<Rx2,Ry1<Ry2

    box_return = [[Lx1,Ly1,Lx2,Ly2],[Rx1,Ry1,Rx2,Ry2]]
    if upDown:
        box_return = [
            [Lx1,Ly1,Lx2,int(Ly1+(Ly2-Ly1)*55/100)],
            [Lx1,int(Ly1+(Ly2-Ly1)*45/100),Lx2,Ly2],
            [Rx1,Ry1,Rx2,int(Ry1+(Ry2-Ry1)*55/100)],
            [Rx1,int(Ry1+(Ry2-Ry1)*45/100),Rx2,Ry2]
            ]
    return box_return

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

def pred(model,img_path,crop,thresh=0.3,iou_s=0.1):
    def restore_coordinate(boxes_s_original,pts):
        boxes_s_restore = []
        for box in boxes_s_original:
            boxes_s_restore.append([int(box[0]+pts[0]),int(box[1]+pts[1]),int(box[2]+pts[0]),int(box[3]+pts[1]),truncate(box[4], 3)])
        return boxes_s_restore

    img_original = cv2.imread(img_path)
    crop = exapnd_upDown_porcess(img_original,crop,expand_lung=True,upDown=False)

    [right_lung_crop,left_lung_crop] = crop
    pts_list = [right_lung_crop,left_lung_crop]
    cls_names = ['WYY','XMZH','XQJY']
    boxesr = []
    clses_final = []
    for pts in pts_list:
        img_crop = img_original[pts[1]:pts[3],pts[0]:pts[2]]
        crop_midY = pts[1]+int((pts[3]-pts[1])/2)
        xmzh_midY = pts[1]+int((pts[3]-pts[1])*3/4)

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
        boxes_s_restore = restore_coordinate(boxes_s_original,pts)
        for i, clse in enumerate(clses_original):
            if clse not in ['WYY','meaningless']:
                # print('boxes_s_restore[i]',boxes_s_restore[i])
                # print('clse[i]',clse)
                y1_i = boxes_s_restore[i][1]
                if clse == 'XQJY' and crop_midY > y1_i:
                    continue
                if clse == 'XMZH' and xmzh_midY < y1_i:
                    continue
                # if clse == 'XQJY':
                    # y1 = boxes_s_restore[i][1]
                boxesr.append(boxes_s_restore[i])
                clses_final.append(clses_original[i])
    return boxesr, clses_final

def vis(img_path,boxesr,clses,crop,output_dir,score=0.1):
    assert len(boxesr) == len(clses)
    [right_lung_crop,left_lung_crop] = crop
    img = cv2.imread(img_path)
    if len(right_lung_crop)>0:
        cv2.rectangle(img,(right_lung_crop[0],right_lung_crop[1]),(right_lung_crop[2],right_lung_crop[3]),(255,0,0),2)
    if len(left_lung_crop)>0:
        cv2.rectangle(img,(left_lung_crop[0],left_lung_crop[1]),(left_lung_crop[2],left_lung_crop[3]),(255,0,0),2)
    for i,box in enumerate(boxesr):
        if box[4] < score:
            continue
        cv2.putText(img, clses[i]+'_'+str(box[4]), (box[0], box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255),2)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir,img_name), img)

if __name__ == '__main__':
    from timeit import default_timer as timer
    import json
    import shutil
    from tqdm import tqdm
    from pycocotools.coco import COCO

    config_file = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_13_XMZH_JY_formal_cfg.py'
    pth_model = '/data/steven/project/Object_Detection_coastal/dr_wrapper/DR_models_configs/dr2_13_XMZH_JY_formal.pth'

    score = 0.3
    start = timer()
    model = init_detector(config_file, pth_model, device='cuda:0')
    elapsed_time = round(timer() - start,2)
    print('{} model loaded {}s {}'.format('-'*20,elapsed_time,'-'*20))
    m1_allocated =  torch.cuda.max_memory_allocated(device='cuda:0')
    m1_cached =  torch.cuda.max_memory_cached(device='cuda:0')
    print('model loaded allocated: {}'.format(byte2readable(m1_allocated)))
    print('model loaded cached: {}'.format(byte2readable(m1_cached)))

    single_test = True
    if single_test:
        # im_name = './test_data/0_single_test_img/PN038159.jpg'
        # im_name = './test_data/0_test_img_original/PN038161.jpg'
        # im_name = '/data/steven/project/Object_Detection_coastal/dataser_raw/0_All_orginal_image_real/orginal_img/PN019926.jpg'

        # im_name = './test_data/banpianying_test/0822/1.3.12.2.1107.5.3.33.4700.11.201901020959500562-1.jpg'
        # crop = [[468, 112, 1403, 2014],[1516, 151, 2269, 2070],[953, 945, 1269, 1477],[1594, 853, 1895, 1302]]
        im_name = '/data/steven/project/Object_Detection_coastal/dr_wrapper/test_data/banpianying_test/0822/1.3.12.2.1107.5.3.33.4700.11.201901020959500562-1.jpg'
        # crop = [[468, 112, 1403, 2014],[1516, 151, 2269, 2070],[953, 945, 1269, 1477],[1594, 853, 1895, 1302]]
        crop = [[10, 10, 2269, 2269],[10, 10, 2269, 2269],[953, 945, 1269, 1477],[1594, 853, 1895, 1302]]

        start = timer()
        boxesr, clses = pred(model,im_name,crop)
        m2_allocated =  torch.cuda.max_memory_allocated(device='cuda:0')
        m2_cached =  torch.cuda.max_memory_cached(device='cuda:0')
        print('model pred allocated: {}'.format(byte2readable(m2_allocated)))
        print('model pred cached: {}'.format(byte2readable(m2_cached)))

        print('_'*100)
        print(boxesr, clses)
        # vis(im_name, boxesr, crop, clses)

        elapsed_time = round(timer() - start,2)
        print('{} model pred {}s {}'.format('-'*20,elapsed_time,'-'*20))
    else:
        import tensorflow as tf
        from dr1_10_8in1_crop_formal import init_sess,find_model
        tfmodel = './DR_models_configs/dr1_10_8in1_crop_formal.ckpt'
        sess, net = init_sess(tfmodel)
        print('Loaded network {:s}'.format(tfmodel))

        # input_dir = '/data/steven/project/Object_Detection_coastal/dataser_raw/1a_AI_pred/images_raw_anno/orginal_img'
        # output_dir = './test_data/2_all_left_test_data_800/2_new_8in1_and_0905_new_banpianying_score_{}_model_with_feimen_filter'.format(str(score))
        # input_dir = './test_data/right_dark_3_img/original'
        # output_dir = './test_data/right_dark_3_img/result_feijiejie'
        input_dir = './test_data/0913_zhuyisheng_test_imgs'
        output_dir = './test_data/0913_zhuyisheng_test_imgs_XMZH_JY'

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        img_list = os.listdir(input_dir)
        vis_img_count = 0
        for img_name in tqdm(img_list):
            img_path = os.path.join(input_dir,img_name)
            output_dict = find_model(sess, net, img_path)
            crop = [output_dict['right_lung_crop'], output_dict['left_lung_crop']]

            if len(crop[0])>0 or len(crop[1])>0:
                boxesr, clses = pred(model,img_path,crop)
                print('boxesr: ', boxesr, 'clses: ', clses)
                if len(boxesr)>0:
                    vis(img_path,boxesr,clses,crop,output_dir,score)
                    vis_img_count += 1
            else:
                '!'*100
                print(img_name, ' has no lung_crop!')
                assert 1>2
        print(vis_img_count)
